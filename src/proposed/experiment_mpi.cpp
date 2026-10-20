/*
  mpic++ experiment_mpi.cpp lib_output.cpp -o experiment_mpi -std=c++20 -O2
  mpirun -np <P> ./experiment_mpi -p prob.txt -d 1e-1 -i 10 -o result.txt

  Design: Coordinator-worker with hash-based leaf assignment.
  - Rank 0 (coordinator): maintains full tree state, runs RD-Tracking logic.
  - Ranks 1..P-1 (workers): receive DRAW requests, sample reward, return result.
  - Leaf assignment: leaf at node_id n is handled by rank (n % P).

  When P=1, all leaves are local to rank 0 (no MPI messages).
*/

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <list>
#include <random>
#include <string>
#include <vector>
#include <mpi.h>
#include "lib_output.h"

// ---------------------------------------------------------------------------
// MPI tags
// ---------------------------------------------------------------------------
constexpr int TAG_DRAW   = 0;  // rank 0 -> worker: int leaf_id
constexpr int TAG_REWARD = 1;  // worker -> rank 0: double reward
constexpr int TAG_STOP   = 2;  // rank 0 -> worker: end of one trial

// ---------------------------------------------------------------------------
// Global RNG (seeded per-rank in main)
// ---------------------------------------------------------------------------
std::mt19937 mt(std::random_device{}());

int bernoulli_draw(double mu) {
  std::uniform_real_distribution<double> u(0.0, 1.0);
  return u(mt) < mu ? 1 : 0;
}

double KLdiv(double p, double q) {
  constexpr double eps = 1e-15;
  p = std::clamp(p, eps, 1.0 - eps);
  q = std::clamp(q, eps, 1.0 - eps);
  return p * std::log(p / q) + (1.0 - p) * std::log((1.0 - p) / (1.0 - q));
}

// ---------------------------------------------------------------------------
// Node types
// ---------------------------------------------------------------------------
enum class NodeType { MAXNODE, MINNODE };
using enum NodeType;

// ---------------------------------------------------------------------------
// Tree_Alg
// ---------------------------------------------------------------------------
class Tree_Alg {
public:
  class Node {
  public:
    int id;
    int depth;
    bool is_leaf;
    NodeType type;
    int parent;
    std::vector<int> child;

    int N;
    double reward_sum;
    double mu_hat;
    double d_value;
    std::vector<double> sub_weight;
    double Z;
  };

  std::vector<Node> nodes;
  std::vector<int> l_to_n;
  std::vector<int> n_to_l;

  inline Node& operator[](int i) { return nodes[i]; }
  int get_leaf_id(int node_id)  { return n_to_l[node_id]; }
  int get_node_id(int leaf_id)  { return l_to_n[leaf_id]; }

  void update_mu_hat(double reward, int leaf_id) {
    int node_id = l_to_n[leaf_id];
    nodes[node_id].reward_sum += reward;
    nodes[node_id].N += 1;
    nodes[node_id].mu_hat = nodes[node_id].reward_sum / (double)nodes[node_id].N;
    node_id = nodes[node_id].parent;
    while (true) {
      Node &node = nodes[node_id];
      if (node.type == MAXNODE) {
        node.mu_hat = 0.0;
        for (int c : node.child) node.mu_hat = std::max(node.mu_hat, nodes[c].mu_hat);
      } else {
        node.mu_hat = 1.0;
        for (int c : node.child) node.mu_hat = std::min(node.mu_hat, nodes[c].mu_hat);
      }
      node_id = node.parent;
      if (node.parent == -1) break;
    }
  }

  void calc_d_value(double theta) {
    constexpr double eps = 1e-5;
    int n = (int)nodes.size();

    if (nodes[0].mu_hat >= theta) {
      // win mode
      for (int i = n - 1; i >= 0; i--) {
        Node &node = nodes[i];
        if (node.is_leaf) {
          node.d_value = node.mu_hat >= theta
            ? KLdiv(std::max(node.mu_hat, theta + eps), theta) : 0.0;
        } else if (node.type == MAXNODE) {
          node.d_value = -1e10;
          for (int c : node.child) node.d_value = std::max(node.d_value, nodes[c].d_value);
        } else {
          bool zero = false; double inv = 0.0;
          for (int c : node.child) {
            if (nodes[c].d_value > 0) inv += 1.0 / nodes[c].d_value;
            else { zero = true; break; }
          }
          node.d_value = zero ? 0.0 : 1.0 / inv;
        }
      }
    } else {
      // lose mode
      for (int i = n - 1; i >= 0; i--) {
        Node &node = nodes[i];
        if (node.is_leaf) {
          node.d_value = node.mu_hat < theta
            ? KLdiv(std::min(node.mu_hat, theta - eps), theta) : 0.0;
        } else if (node.type == MINNODE) {
          node.d_value = -1e10;
          for (int c : node.child) node.d_value = std::max(node.d_value, nodes[c].d_value);
        } else {
          bool zero = false; double inv = 0.0;
          for (int c : node.child) {
            if (nodes[c].d_value > 0) inv += 1.0 / nodes[c].d_value;
            else { zero = true; break; }
          }
          node.d_value = zero ? 0.0 : 1.0 / inv;
        }
      }
    }
  }

  void calc_Z(double theta) {
    int n = (int)nodes.size();

    if (nodes[0].mu_hat >= theta) {
      for (int i = n - 1; i >= 0; i--) {
        Node &node = nodes[i];
        if (node.is_leaf) {
          node.Z = node.mu_hat >= theta ? node.N * KLdiv(node.mu_hat, theta) : 0.0;
        } else if (node.type == MAXNODE) {
          node.Z = 0.0;
          for (int c : node.child) node.Z += nodes[c].Z;
        } else {
          node.Z = 1e100;
          for (int c : node.child) node.Z = std::min(node.Z, nodes[c].Z);
        }
      }
    } else {
      for (int i = n - 1; i >= 0; i--) {
        Node &node = nodes[i];
        if (node.is_leaf) {
          node.Z = node.mu_hat < theta ? node.N * KLdiv(node.mu_hat, theta) : 0.0;
        } else if (node.type == MINNODE) {
          node.Z = 0.0;
          for (int c : node.child) node.Z += nodes[c].Z;
        } else {
          node.Z = 1e100;
          for (int c : node.child) node.Z = std::min(node.Z, nodes[c].Z);
        }
      }
    }
  }
};

// ---------------------------------------------------------------------------
// Tree_Prob  (shape + true parameters)
// ---------------------------------------------------------------------------
class Tree_Prob {
public:
  class Node {
  public:
    int id, depth;
    bool is_leaf;
    NodeType type;
    int parent;
    std::vector<int> child;
    double true_mu;
  };

  std::vector<Node> nodes;
  std::vector<int> l_to_n;
  std::vector<int> n_to_l;

  inline Node& operator[](int i) { return nodes[i]; }
  inline const Node& operator[](int i) const { return nodes[i]; }
  int get_node_id(int leaf_id) const { return l_to_n[leaf_id]; }
};

// ---------------------------------------------------------------------------
// Problem
// ---------------------------------------------------------------------------
class Problem {
public:
  int N_size, L_size;
  Tree_Prob tree;
  double theta;
  bool is_win;

  void read_probfile(const std::string &filename) {
    std::ifstream ifs(filename);
    if (!ifs) { std::cerr << "error: cannot open " << filename << "\n"; std::exit(1); }
    std::string tmp;

    ifs >> tmp >> N_size;
    tree.nodes.resize(N_size);

    ifs >> tmp >> theta >> tmp;
    is_win = (tmp == "win");

    ifs >> tmp;  // [nodes]
    std::list<int> l_to_n_tmp;
    tree.n_to_l.resize(N_size);
    int leaf_count = 0;

    for (int id = 0; id < N_size; id++) {
      ifs >> tree[id].id >> tree[id].depth >> tmp;
      if (tmp == "true") {
        tree[id].is_leaf = true;
        tree.n_to_l[id] = leaf_count++;
        l_to_n_tmp.push_back(id);
      } else {
        tree[id].is_leaf = false;
        tree.n_to_l[id] = -1;
      }
      ifs >> tmp;
      tree[id].type = (tmp == "MAX") ? MAXNODE : MINNODE;
      ifs >> tree[id].true_mu >> tree[id].parent;
      int cc; ifs >> cc;
      tree[id].child.resize(cc);
      for (int c = 0; c < cc; c++) ifs >> tree[id].child[c];
    }
    L_size = leaf_count;
    tree.l_to_n = std::vector<int>(l_to_n_tmp.begin(), l_to_n_tmp.end());

    ifs >> tmp;
    if (tmp != "end") { std::cerr << "error: expected 'end'\n"; std::exit(1); }
    ifs.close();
  }
};

// ---------------------------------------------------------------------------
// calc_weight  (same as origin)
// ---------------------------------------------------------------------------
void calc_weight(Tree_Alg &tree, double theta) {
  int N_size = (int)tree.nodes.size();
  int L_size = (int)tree.nodes[0].sub_weight.size();

  if (tree[0].mu_hat >= theta) {
    // win mode
    for (int i = N_size - 1; i >= 0; i--) {
      Tree_Alg::Node &node = tree[i];
      if (node.is_leaf) {
        node.sub_weight.assign(L_size, 0.0);
        node.sub_weight[tree.n_to_l[i]] = 1.0;
      } else if (node.d_value > 0) {
        if (node.type == MAXNODE) {
          int best = 0; double bd = tree[node.child[0]].d_value;
          for (int c = 1; c < (int)node.child.size(); c++) {
            if (tree[node.child[c]].d_value > bd) { bd = tree[node.child[c]].d_value; best = c; }
          }
          node.sub_weight = tree[node.child[best]].sub_weight;
        } else {
          node.sub_weight.assign(L_size, 0.0);
          double inv = 0.0;
          for (int c : node.child) inv += 1.0 / tree[c].d_value;
          for (int c : node.child)
            for (int l = 0; l < L_size; l++)
              node.sub_weight[l] += tree[c].sub_weight[l] / tree[c].d_value / inv;
        }
      } else {
        node.sub_weight.assign(L_size, 1.0 / L_size);
      }
    }
  } else {
    // lose mode
    for (int i = N_size - 1; i >= 0; i--) {
      Tree_Alg::Node &node = tree[i];
      if (node.is_leaf) {
        node.sub_weight.assign(L_size, 0.0);
        node.sub_weight[tree.n_to_l[i]] = 1.0;
      } else if (node.d_value > 0) {
        if (node.type == MINNODE) {
          int best = 0; double bd = tree[node.child[0]].d_value;
          for (int c = 1; c < (int)node.child.size(); c++) {
            if (tree[node.child[c]].d_value > bd) { bd = tree[node.child[c]].d_value; best = c; }
          }
          node.sub_weight = tree[node.child[best]].sub_weight;
        } else {
          node.sub_weight.assign(L_size, 0.0);
          double inv = 0.0;
          for (int c : node.child) inv += 1.0 / tree[c].d_value;
          for (int c : node.child)
            for (int l = 0; l < L_size; l++)
              node.sub_weight[l] += tree[c].sub_weight[l] / tree[c].d_value / inv;
        }
      } else {
        node.sub_weight.assign(L_size, 1.0 / L_size);
      }
    }
  }
}

// ---------------------------------------------------------------------------
// MPI broadcast helper
// ---------------------------------------------------------------------------
void broadcast_prob(Problem &prob, int mpi_rank) {
  // 1. scalar fields
  struct Scalars { int N_size, L_size, is_win; double theta; };
  Scalars sc{};
  if (mpi_rank == 0) sc = {prob.N_size, prob.L_size, prob.is_win ? 1 : 0, prob.theta};
  MPI_Bcast(&sc, sizeof(Scalars), MPI_BYTE, 0, MPI_COMM_WORLD);
  if (mpi_rank != 0) {
    prob.N_size = sc.N_size;
    prob.L_size = sc.L_size;
    prob.theta  = sc.theta;
    prob.is_win = sc.is_win != 0;
    prob.tree.nodes.resize(prob.N_size);
    prob.tree.l_to_n.resize(prob.L_size);
    prob.tree.n_to_l.resize(prob.N_size);
  }

  // 2. l_to_n, n_to_l
  MPI_Bcast(prob.tree.l_to_n.data(), prob.L_size, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(prob.tree.n_to_l.data(), prob.N_size, MPI_INT, 0, MPI_COMM_WORLD);

  // 3. node data (variable-length children -> serialize to flat arrays)
  std::vector<int> idata;    // id, depth, is_leaf, type, parent, child_count, child...
  std::vector<double> ddata; // true_mu per node

  if (mpi_rank == 0) {
    for (int i = 0; i < prob.N_size; i++) {
      auto &n = prob.tree.nodes[i];
      idata.push_back(n.id);
      idata.push_back(n.depth);
      idata.push_back(n.is_leaf ? 1 : 0);
      idata.push_back(n.type == MAXNODE ? 0 : 1);
      idata.push_back(n.parent);
      idata.push_back((int)n.child.size());
      for (int c : n.child) idata.push_back(c);
      ddata.push_back(n.true_mu);
    }
  }

  int int_sz = (int)idata.size();
  MPI_Bcast(&int_sz, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (mpi_rank != 0) { idata.resize(int_sz); ddata.resize(prob.N_size); }
  MPI_Bcast(idata.data(), int_sz, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(ddata.data(), prob.N_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  if (mpi_rank != 0) {
    int idx = 0;
    for (int i = 0; i < prob.N_size; i++) {
      auto &n = prob.tree.nodes[i];
      n.id     = idata[idx++];
      n.depth  = idata[idx++];
      n.is_leaf = idata[idx++] != 0;
      n.type   = (idata[idx++] == 0) ? MAXNODE : MINNODE;
      n.parent = idata[idx++];
      int cc   = idata[idx++];
      n.child.resize(cc);
      for (int c = 0; c < cc; c++) n.child[c] = idata[idx++];
      n.true_mu = ddata[i];
    }
  }
}

// ---------------------------------------------------------------------------
// Oracle  (rank 0 only: N tracking + local sampling)
// ---------------------------------------------------------------------------
class Oracle {
  int L_size;
  std::vector<double> true_mu;
  int N_total = 0;

public:
  Oracle() = default;
  explicit Oracle(const Problem &prob) : L_size(prob.L_size), true_mu(prob.L_size) {
    for (int i = 0; i < L_size; i++)
      true_mu[i] = prob.tree.nodes[prob.tree.l_to_n[i]].true_mu;
  }

  // Local draw (leaf assigned to rank 0)
  double draw(int leaf_id) {
    N_total++;
    return bernoulli_draw(true_mu[leaf_id]);
  }

  // Track a draw that happened on another rank
  void track() { N_total++; }

  int get_N_total() const { return N_total; }
  void reset_N() { N_total = 0; }
};

// ---------------------------------------------------------------------------
// Args
// ---------------------------------------------------------------------------
class Args {
public:
  std::string probfile = "prob.txt";
  double delta_exp = -1.0, delta_coef = 1.0;
  int itr_num = 100;
  std::string outfile = "result.txt";
  bool silent = false;
  double theta = -1.0;

  void init(int argc, char **argv) {
    int i = 1;
    while (i < argc) {
      std::string a = argv[i];
      if (a == "-p" || a == "--prob") {
        probfile = argv[i + 1]; i += 2;
      } else if (a == "-d" || a == "--delta") {
        std::string s = argv[i + 1];
        auto pos = s.find_first_of("eE");
        delta_coef = std::stod(s.substr(0, pos));
        delta_exp  = std::stod(s.substr(pos + 1));
        i += 2;
      } else if (a == "-i" || a == "--itr") {
        itr_num = std::atoi(argv[i + 1]); i += 2;
      } else if (a == "-o" || a == "--out") {
        outfile = argv[i + 1]; i += 2;
      } else if (a == "--silent") {
        silent = true; i += 1;
      } else if (a == "-t" || a == "--theta") {
        theta = std::stod(argv[i + 1]); i += 2;
      } else {
        std::cerr << "error: unknown argument [" << a << "]\n";
        std::exit(1);
      }
    }
  }
};

// ---------------------------------------------------------------------------
// Alg_WN_Dtracking_MPI
// ---------------------------------------------------------------------------
class Alg_WN_Dtracking_MPI {
  int N_size, L_size;
  Tree_Alg tree;
  double theta, delta_coef, delta_exp;

public:
  void set_param(const Problem &prob, double dc, double de) {
    N_size = prob.N_size;
    L_size = prob.L_size;
    tree.nodes.resize(N_size);
    tree.l_to_n = prob.tree.l_to_n;
    tree.n_to_l = prob.tree.n_to_l;
    for (int i = 0; i < N_size; i++) {
      auto &n = tree[i];
      n.id     = prob.tree[i].id;
      n.depth  = prob.tree[i].depth;
      n.is_leaf = prob.tree[i].is_leaf;
      n.type   = prob.tree[i].type;
      n.parent = prob.tree[i].parent;
      n.child  = prob.tree[i].child;
    }
    theta = prob.theta;
    delta_coef = dc;
    delta_exp  = de;
  }

  void reset() {
    for (int i = 0; i < N_size; i++) {
      tree[i].N = 0; tree[i].reward_sum = 0;
      tree[i].mu_hat = 0; tree[i].d_value = 0;
      tree[i].sub_weight.assign(L_size, 0.0);
      tree[i].Z = 0;
    }
  }

  double beta(int t) const {
    return std::log(std::log(t + 1.0))
           - std::log(delta_coef)
           - delta_exp * std::log(10.0);
  }

  bool is_stop(int t) const {
    double b = beta(t);
    return tree.nodes[0].Z >= b || -tree.nodes[0].Z >= b;
  }

  // -------------------------------------------------------------------
  // Coordinator (rank 0): run one trial, return win/lose
  // -------------------------------------------------------------------
  bool run_root(Oracle &oracle, int mpi_size) {
    reset();

    // --- Forced init: draw each leaf once ---
    for (int ell = 0; ell < L_size; ell++) {
      double reward = do_draw(ell, oracle, mpi_size);
      tree.update_mu_hat(reward, ell);
    }

    // --- Main loop ---
    int t = 1;
    while (true) {
      // Forced exploration: find least-sampled leaf
      int least_leaf = 0, least_count = tree[tree.l_to_n[0]].N;
      for (int l = 1; l < L_size; l++) {
        int nl = tree[tree.l_to_n[l]].N;
        if (nl < least_count) { least_count = nl; least_leaf = l; }
      }

      int next_leaf;
      double thr = std::sqrt((double)t) - L_size / 2.0;
      if (least_count < thr) {
        next_leaf = least_leaf;
      } else {
        tree.calc_d_value(theta);
        calc_weight(tree, theta);
        const auto &w = tree.nodes[0].sub_weight;
        double best = -1e100;
        next_leaf = 0;
        for (int l = 0; l < L_size; l++) {
          double Nl = (double)tree[tree.l_to_n[l]].N;
          double ratio = w[l] / Nl;
          if (ratio > best) { best = ratio; next_leaf = l; }
        }
      }

      double reward = do_draw(next_leaf, oracle, mpi_size);
      tree.update_mu_hat(reward, next_leaf);
      tree.calc_Z(theta);

      if (is_stop(t)) {
        // Tell all workers this trial is over
        int dummy = 0;
        for (int r = 1; r < mpi_size; r++)
          MPI_Send(&dummy, 1, MPI_INT, r, TAG_STOP, MPI_COMM_WORLD);
        return tree.nodes[0].mu_hat >= theta;
      }
      t++;
    }
  }

  // -------------------------------------------------------------------
  // Worker: handle one trial (loop until TAG_STOP)
  // -------------------------------------------------------------------
  static void run_worker(const std::vector<double> &true_mu) {
    while (true) {
      int leaf_id;
      MPI_Status status;
      MPI_Recv(&leaf_id, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
      if (status.MPI_TAG == TAG_STOP) break;
      double reward = bernoulli_draw(true_mu[leaf_id]);
      MPI_Send(&reward, 1, MPI_DOUBLE, 0, TAG_REWARD, MPI_COMM_WORLD);
    }
  }

private:
  // Execute a draw: local or remote depending on hash assignment
  double do_draw(int leaf_id, Oracle &oracle, int mpi_size) {
    int node_id = tree.l_to_n[leaf_id];
    int target  = node_id % mpi_size;
    if (target == 0) {
      return oracle.draw(leaf_id);
    } else {
      MPI_Send(&leaf_id, 1, MPI_INT, target, TAG_DRAW, MPI_COMM_WORLD);
      double reward;
      MPI_Recv(&reward, 1, MPI_DOUBLE, target, TAG_REWARD, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      oracle.track();
      return reward;
    }
  }
};

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int mpi_rank, mpi_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

  // Per-process RNG seed
  mt.seed(std::random_device{}() + (unsigned)mpi_rank * 999983u);

  // Parse args on rank 0, then broadcast key scalars
  Args args;
  if (mpi_rank == 0) args.init(argc, argv);

  struct ArgsBC { int itr_num; double delta_coef, delta_exp, theta; };
  ArgsBC abc{};
  if (mpi_rank == 0) abc = {args.itr_num, args.delta_coef, args.delta_exp, args.theta};
  MPI_Bcast(&abc, sizeof(ArgsBC), MPI_BYTE, 0, MPI_COMM_WORLD);
  if (mpi_rank != 0) {
    args.itr_num    = abc.itr_num;
    args.delta_coef = abc.delta_coef;
    args.delta_exp  = abc.delta_exp;
    args.theta      = abc.theta;
  }

  // Read and broadcast problem
  Problem prob;
  if (mpi_rank == 0) {
    prob.read_probfile(args.probfile);
    if (args.theta > 0.0) prob.theta = args.theta;
  }
  broadcast_prob(prob, mpi_rank);
  // Apply theta override on workers too (theta is already in prob after broadcast,
  // but if it was overridden before broadcast it's already correct)

  // Build true_mu array for workers
  std::vector<double> true_mu(prob.L_size);
  for (int i = 0; i < prob.L_size; i++)
    true_mu[i] = prob.tree.nodes[prob.tree.l_to_n[i]].true_mu;

  // -------------------------------------------------------------------
  // Coordinator path (rank 0)
  // -------------------------------------------------------------------
  if (mpi_rank == 0) {
    Timer timer;
    std::string start_date = timer.now_str();

    Alg_WN_Dtracking_MPI alg;
    alg.set_param(prob, args.delta_coef, args.delta_exp);
    Oracle oracle(prob);

    timer.start();

    double total_tau = 0.0, total_tau2 = 0.0;
    int correct_count = 0;
    Timer log_timer;
    log_timer.start();

    for (int itr = 1; itr <= args.itr_num; itr++) {
      bool ans = alg.run_root(oracle, mpi_size);
      if (ans == prob.is_win) correct_count++;
      double tau = oracle.get_N_total();
      total_tau  += tau;
      total_tau2 += tau * tau;
      oracle.reset_N();

      if (!args.silent && log_timer.elapsed_sec() >= 10) {
        std::cout << timer.now_str() << " " << itr << "/" << args.itr_num << "\n";
        log_timer.start();
      }
    }

    double tau_avg = total_tau / args.itr_num;
    double tau_var = total_tau2 / args.itr_num - tau_avg * tau_avg;
    tau_var = std::max(tau_var, 0.0);

    std::ofstream ofs(args.outfile);
    ofs << "start: "         << start_date << "\n";
    ofs << "finish: "        << timer.now_str() << " (" << timer.elapsed_str() << ")\n";
    ofs << "probfile: "      << args.probfile << "\n";
    ofs << "theta: "         << prob.theta << "\n";
    ofs << "delta: "         << args.delta_coef << "e" << args.delta_exp << "\n";
    ofs << "algorithm: Ratio-based D-Tracking (MPI)\n";
    ofs << "mpi_size: "      << mpi_size << "\n";
    ofs << "itr: "           << args.itr_num << "\n";
    ofs << "correct_count: " << correct_count << "\n";
    ofs << "tau_avg: "       << tau_avg << "\n";
    ofs << "tau_var: "       << tau_var << "\n";
    ofs << "tau_stddev: "    << std::sqrt(tau_var) << "\n";
    ofs.close();

    std::cout << "finish\n-> " << args.outfile << "\n";
  }
  // -------------------------------------------------------------------
  // Worker path (rank != 0)
  // -------------------------------------------------------------------
  else {
    for (int itr = 0; itr < args.itr_num; itr++) {
      Alg_WN_Dtracking_MPI::run_worker(true_mu);
    }
  }

  MPI_Finalize();
  return 0;
}
