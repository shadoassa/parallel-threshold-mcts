/*
  g++ experiment.cpp -o experiment -std=c++20 -O2
  ./experiment prob1.txt 0.1 Proposed_method 100 result.txt
*/


// 
// [実験設定]
// 
enum class Experiment {SAMPLING_PROPORTION, STOPPING_TIME};

// 進捗表示間隔
int log_span_sec = 10;

bool record_each_iteration = true;

/*
  実験1: 選択割合の収束

  ・stopping conditionを以下に差し替え:
    true if t > T_end
    false if t <= T_end

  ・t % t_step = 0 を満たす各時刻で、
    選択割合の誤差 max_ell |w^*_ell - N_ell/t| を計算、記録する

  ・出力ファイル内容:
    ・T_endとT_stepの値
    ・各記録時刻における選択割合の誤差の平均と分散
*/
int T_END = 10000;
int T_STEP = 100;

/*
  実験2: 停止時刻

  停止時刻の平均と分散を出力
*/
bool show_draw_ratio_when_stop = false;
bool disable_forced_exploration = false;
bool stop_with_fixed_t = false;

bool adjust_theta_leaf = false;
enum class WeightType {RANDOM, UNIFORM};
WeightType weight_otherwise = WeightType::UNIFORM;



#include <algorithm>
#include <cmath>
#include <cfloat> // DBL_MAX
#include <fstream>
#include <iomanip>
#include <iostream>
#include <list>
#include <map>
#include <random>
#include <string>
#include <vector>
#include <limits.h>
#include <set>
#include <unordered_set>

#include <mpi.h> // OPEN MPI

#include <boost/random/beta_distribution.hpp>

#include "lib_output.h"






enum class NodeType {MAXNODE, MINNODE};
std::vector<std::string> NodeType_str = {"MaxNode", "MinNode"};
using enum NodeType;
std::ostream& operator<<(std::ostream& os, const NodeType a){
  os << NodeType_str[static_cast<int>(a)];
  return os;
}



// enum class StopLabel {HIGH, LOW, UNCERTAIN};
// std::vector<std::string> StopLabel_str = {"HIGH", "LOW", "UNCERTAIN"};
// // using enum StopLabel;
// std::ostream& operator<<(std::ostream& os, const StopLabel a){
//   os << StopLabel_str[static_cast<int>(a)];
//   return os;
// }

enum class Alg_enum {
  D_TRACKING, 
  C_TRACKING, 
  RATIO_D_TRACKING, 
  P_TRACKING, 
  RANDOM_TRACKING, 
  UGAPE_MCTS, 
  LUCB_MICRO, 
  ROUND_ROBIN};
std::vector<std::string> Alg_enum_str = {
  "D-Tracking", 
  "C-Tracking", 
  "Ratio-based D-Tracking", 
  "P-Tracking", 
  "Random-Tracking", 
  "UGapE-MCTS", 
  "LUCB-micro", 
  "Round Robin"};
using enum Alg_enum;
std::ostream& operator<<(std::ostream& os, const Alg_enum a){
  os << Alg_enum_str[static_cast<int>(a)];
  return os;
}

bool had_break = false;

void write_break(const std::string msg = ""){
  std::ofstream ofs("break.txt", std::ios::out);
  ofs << "break!" << msg << std::endl;
  ofs.close();
}

// デバッグ用ブレークポイント関数
void breakfunc_label(){};
void breakfunc(const std::string msg = ""){
  std::ofstream ofs("break.txt", std::ios::out);
  ofs << "break!" << msg << std::endl;
  ofs.close();
  std::cout << "break! " << msg << std::endl;
  breakfunc_label();
  // exit(1);
}
void breakfunc_if(const bool cond, const std::string msg = ""){
  if(cond){
    std::ofstream ofs("break.txt", std::ios::out);
    ofs << "break!" << msg << std::endl;
    ofs.close();
    std::cout << "break! " << msg << std::endl;
    breakfunc_label();
    exit(1);
  }
}
std::unordered_set<std::string> break_flag_set = {};
void breakfunc_once(std::string label, const std::string msg = ""){
  if(!break_flag_set.contains(label)){
    std::cout << "break with \"" << label << "\"! " << msg << std::endl;
    break_flag_set.insert(label);
    breakfunc_label();
  }
}

int debug_count_1 = 0;
int debug_count_2 = 0;
int debug_count_3 = 0;


std::random_device rnd_dev;
// std::mt19937 mt((int)time(0));
std::mt19937 mt(rnd_dev());

// 整数乱数生成 閉区間[a,b]
int randRange(const int a, const int b){
  if(a > b){
    breakfunc("randRange error");
    return -1;
  }
  else if(a == b) return b;
  else{
    std::uniform_int_distribution<> unirand(a,b);
    return(unirand(mt));
  }
}

// ベルヌーイ分布から抽出
int bernoulli(const double mu){
  std::uniform_real_distribution<> unirand(0,1);
  if(unirand(mt) >= mu) return 0;
  else return 1;
}

double KLdiv(double p, double q){
  // 一応クリッピング
  double epsilon = 1e-15;
  if(p < epsilon) p = epsilon;
  else if(p > 1-epsilon) p = 1-epsilon;
  if(q < epsilon) q = epsilon;
  else if(q > 1-epsilon) q = 1-epsilon;

  return p * std::log(p/q) + (1-p) * std::log((1-p)/(1-q));
}



struct LeafID{
  int val;
};



class Algorithm_base;

class Args{
  public:
  std::string probfile;
  // double delta; // 計算誤差軽減のため削除
  // delta = delta_coef * 10^{delta_exp}とする
  // log(1/delta) = - log(delta_coef) - delta_exp * log(10)
  double delta_exp; // 指数部
  double delta_coef; // 仮数部
  Alg_enum algorithm;
  int itr_num;
  std::string outfile;
  std::string outfile2;
  Experiment exp;
  bool silent;
  bool enable_second_output;

  double theta;
  /*
    -i --infile : string
    -d --delta : double
    -a --algorithm : choice {"Propose_method", "C-tracking", ...} \cup {"0","1","2",...}
    -n --num : int
    -o --outfile : string
  */

  void set_default(){
    this->probfile = "prob.txt";
    // this->delta = 0.1;
    this->delta_exp = -1;
    this->delta_coef = 1;
    this->algorithm = D_TRACKING;
    this->itr_num = 100;
    this->outfile = "result.txt";
    this->outfile2 = "None";
    this->enable_second_output = false;
    this->exp = Experiment::STOPPING_TIME;
    this->silent = false;
    this->theta = -1;
  }

  void init(int argc, char **argv){
    this->set_default();

    int i = 1;
    while(true){
      if(i >= argc) break;
      // std::cout << argv[i] << std::endl;
      std::string tmp = argv[i];

      if(tmp == "-p" || tmp == "--prob"){
        this->probfile = argv[i+1];
        i += 2;
      }
      else if(tmp == "-d" || tmp == "--delta"){
        std::string str = argv[i+1];
        std::size_t pos = str.find_first_of("eE");
        std::string coef_str = str.substr(0, pos);
        std::string exp_str = str.substr(pos + 1);
        this->delta_coef = std::stod(coef_str);
        this->delta_exp = std::stod(exp_str);

        // this->delta = std::atof(argv[i+1]);
        i += 2;
      }
      else if(tmp == "-a" || tmp == "--algorithm"){
        std::string algname = argv[i+1];
        if(algname == "D-Tracking" || algname == "D"){
          this->algorithm = D_TRACKING;
        }
        else if(algname == "C-Tracking" || algname == "C"){
          this->algorithm = C_TRACKING;
        }
        else if(algname == "P-Tracking" || algname == "P"){
          this->algorithm = P_TRACKING;
        }
        else if(algname == "Random-Tracking" || algname == "Random"){
          this->algorithm = RANDOM_TRACKING;
        }
        else if(algname == "Ratio-based-D-Tracking" || algname == "Ratio"){
          this->algorithm = RATIO_D_TRACKING;
        }
        else if(algname == "UGapE-MCTS" || algname == "UGapE"){
          this->algorithm = UGAPE_MCTS;
        }
        else if(algname == "LUCB-micro" || algname == "LUCB"){
          this->algorithm = LUCB_MICRO;
        }
        else if(algname == "Round-Robin" || algname == "RR"){
          this->algorithm = ROUND_ROBIN;
        }
        else{
          std::cout << "error! args: unknown algorithm_name \"" << algname << "\"" << std::endl;
          exit(1);
        }
        i += 2;
      }
      else if(tmp == "-i" || tmp == "--itr"){
        this->itr_num = std::atoi(argv[i+1]);
        i += 2;
      }
      else if(tmp == "-o" || tmp == "--out"){
        this->outfile = argv[i+1];
        i += 2;
      }
      else if(tmp == "-o2" || tmp == "--out2"){
        this->outfile2 = argv[i+1];
        this->enable_second_output = true;
        i += 2;
      }
      else if(tmp == "--silent"){
        this->silent = true;
        i += 1;
      }
      else if(tmp == "-e" || tmp == "--experiment"){
        std::string expname = argv[i+1];
        if(expname == "1"){
          this->exp = Experiment::SAMPLING_PROPORTION;
        }
        else if(expname == "2"){
          this->exp = Experiment::STOPPING_TIME;
        }
        else{
          std::cout << "error! args: unknown experiment_name" << std::endl;
        }
        i += 2;
      }
      else if(tmp == "-t" || tmp == "--theta"){
        this->theta = std::stod(argv[i+1]);
        i += 2;
      }
      else{
        std::cout << "error! args: unknown argument [" << tmp << "]" << std::endl;
        exit(1);
        // i += 1;
      }
    }
  }

  void print() const {
    std::cout << "probfile: " << this->probfile << "\n";
    std::cout << "delta: " << this->delta_coef << "e" << this->delta_exp << "\n";
    std::cout << "algorithm: " << this->algorithm << "\n";
    std::cout << "itr: " << this->itr_num << "\n";
    std::cout << "outfile: " << this->outfile << "\n";
    std::cout << std::endl;
  }
};

/*
  [TreeShape]
  ノードid
  depth
  is_leaf
  MIN/MAX
  親id
  vector<子id>


  [追加情報_hidden]
  真の期待評価値 / minmax-value
  // 真のd-value


  [追加情報_アルゴリズム]
  訪問回数 (葉のみ)
  平均評価値 / そのMINIMAX (全ノード)
  推定d-value (全ノード)
*/
class Problem;



// githubに公開される前に書き直したいけど時間がないので後でやりますという宣言だけ
// class Tree_Shape{
//   class Node{
//     public:
//     NodeID node_id;
//     LeafID leaf_id; // 非leafなら-1
//     int depth;
//     bool is_leaf;
//     NodeType type;
//     int parent;
//     std::vector<int> child;

//     void print(){
//       std::cout << "node_id: " << this->node_id << "\n";
//       std::cout << "leaf_id: " << this->leaf_id << "\n";
//       std::cout << "depth: " << this->depth << "\n";
//       std::cout << "is_leaf: " << this->is_leaf << "\n";
//       std::cout << "type: " << this->type << "\n";
//       std::cout << "parent: " << this->parent << "\n";
//       std::cout << "child: " << this->child << "\n";
//       std::cout << std::endl;
//     }
//   };

//   std::vector<Node> nodes;
//   std::vector<int> l_to_n; // l_to_n[leaf_id] -> 対応するnode_id
//   std::vector<int> n_to_l; // n_to_l[node_id] -> 対応するleaf_id

// };

class Tree_Alg{
  public:
  class Node{
    public:
    // Probと共通
    int id;
    int depth;
    bool is_leaf;
    NodeType type;
    int parent;
    std::vector<int> child;

    // Alg特有
    int N; // num_visit
    double reward_sum; 
    double mu_hat; // V_s
    double d_value;
    std::vector<double> sub_weight;
    double Z; // Z^\text{upper}_s
    // double weight;

    void print(){
      std::cout << "node_id: " << this->id << "\n";
      std::cout << "depth: " << this->depth << "\n";
      std::cout << "is_leaf: " << this->is_leaf << "\n";
      std::cout << "type: " << this->type << "\n";
      std::cout << "parent: " << this->parent << "\n";
      std::cout << "child: " << this->child << "\n";
      std::cout << "N: " << this->N << "\n";
      std::cout << "reward_sum: " << this->reward_sum << "\n";
      std::cout << "mu_hat: " << this->mu_hat << "\n";
      std::cout << "d_value: " << this->d_value << "\n";
      std::cout << "sub_weight: " << this->sub_weight << "\n";
      std::cout << "Z: " << this->Z << "\n";
      std::cout << std::endl;
    }
  };

  // int N_size;
  // int L_size;
  // int leaf_index_offset;
  std::vector<Node> nodes;

  std::vector<int> l_to_n; // l_to_n[leaf_id] -> 対応するnode_id
  std::vector<int> n_to_l; // n_to_l[node_id] -> 対応するleaf_id

  int get_leaf_id(int node_id){
    return n_to_l[node_id];
  }
  int get_node_id(int leaf_id){
    return l_to_n[leaf_id];
  }

  inline Node& operator[](int i){
    return this->nodes[i];
  }

  void print_mu_hat(){
    for(auto x : nodes){
      std::cout << x.mu_hat << ", ";
    }
    std::cout << std::endl;
  }

  void print_d_value(){
    for(auto x : nodes){
      std::cout << x.d_value << ", ";
    }
    std::cout << std::endl;
  }

  void print(){
    // std::cout << "N_size: " << this->N_size << "\n";
    // std::cout << "L_size: " << this->L_size << "\n";
    std::cout << "l_to_n: " << this->l_to_n << "\n";
    std::cout << "n_to_l: " << this->n_to_l << "\n";
    // std::cout << "offset: " << this->leaf_index_offset << "\n";
    // nodes
    std::cout << std::endl;
  }

  void calc_mu_hat(){
    // nodesは浅い順に格納されているので、逆順なら葉から順に処理できる
    for(int i=nodes.size()-1; i>=0; i--){
      if(nodes[i].is_leaf){
        // pass
      }
      else if(nodes[i].type == MAXNODE){
        double max_mu_hat = nodes[nodes[i].child[0]].mu_hat;
        for(int c : nodes[i].child){
          double mu_hat = nodes[c].mu_hat;
          if(max_mu_hat < mu_hat) max_mu_hat = mu_hat;
        }
        nodes[i].mu_hat = max_mu_hat;
      }
      else if(nodes[i].type == MINNODE){
        double min_mu_hat = nodes[nodes[i].child[0]].mu_hat;
        for(int c : nodes[i].child){
          double mu_hat = nodes[c].mu_hat;
          if(min_mu_hat > mu_hat) min_mu_hat = mu_hat;
        }
        nodes[i].mu_hat = min_mu_hat;
      }
    }
  }

  void update_mu_hat(double reward, int leaf_id){
    int node_id = this->get_node_id(leaf_id);
    nodes[node_id].reward_sum += reward;
    nodes[node_id].N += 1;
    nodes[node_id].mu_hat = nodes[node_id].reward_sum / static_cast<double>(nodes[node_id].N);
    node_id = nodes[node_id].parent;
    while(true){
      Tree_Alg::Node &node = nodes[node_id];
      if(node.type == MAXNODE){
        node.mu_hat = 0;
        for(int c: node.child){
          node.mu_hat = std::max(node.mu_hat, nodes[c].mu_hat);
        }
      }
      else{
        node.mu_hat = 1;
        for(int c: node.child){
          node.mu_hat = std::min(node.mu_hat, nodes[c].mu_hat);
        }
      }
      node_id = node.parent;
      if(node.parent == -1) break;
    }
  }

  // V_{s_0}は計算済みとする
  void calc_d_value(double theta){
    double epsilon = 1e-5;
    int N_size = nodes.size();

    // a_{s_0}(\hat{\bm{\mu}}) = 'win'
    if(nodes[0].mu_hat >= theta){
      for(int i=N_size-1; i>=0; i--){
        Tree_Alg::Node &node = nodes[i];
        if(node.is_leaf == true){
          if(node.mu_hat >= theta){
            // node.d_value = KLdiv(node.mu_hat, theta);
            node.d_value = KLdiv(std::max(node.mu_hat, theta + epsilon), theta);
          }
          else{
            node.d_value = 0;
          }
        }
        else if(node.type == MAXNODE){
          node.d_value = -1e10;
          for(int c: node.child){
            node.d_value = std::max(node.d_value, nodes[c].d_value);
          }
        }
        else if(node.type == MINNODE){
          bool zero_flag = false;
          double sum_d_inv = 0;
          for(int c: node.child){
            double d = nodes[c].d_value;
            if(d > 0){
              sum_d_inv += 1.0 / d;
            }
            else{
              zero_flag = true;
              break;
            }
          }
          if(zero_flag == true){
            node.d_value = 0;
          }
          else{
            node.d_value = 1.0 / sum_d_inv;
          }
        }
      }
    }
    // a_{s_0}(\hat{\bm{\mu}}) = 'lose'
    else{
      for(int i=N_size-1; i>=0; i--){
        Tree_Alg::Node &node = nodes[i];
        if(node.is_leaf == true){
          if(node.mu_hat < theta){
            // node.d_value = KLdiv(node.mu_hat, theta);
            node.d_value = KLdiv(std::min(node.mu_hat, theta - epsilon), theta);
          }
          else{
            node.d_value = 0;
          }
        }
        else if(node.type == MINNODE){
          node.d_value = -1e10;
          for(int c: node.child){
            node.d_value = std::max(node.d_value, nodes[c].d_value);
          }
        }
        else if(node.type == MAXNODE){
          bool zero_flag = false;
          double sum_d_inv = 0;
          for(int c: node.child){
            double d = nodes[c].d_value;
            if(d > 0){
              sum_d_inv += 1.0 / d;
            }
            else{
              zero_flag = true;
              break;
            }
          }
          if(zero_flag == true){
            node.d_value = 0;
          }
          else{
            node.d_value = 1.0 / sum_d_inv;
          }
        }
      }
    }
  }

  void calc_Z(double theta){
    double epsilon = 1e-5;
    int N_size = nodes.size();

    // a_{s_0}(\hat{\bm{\mu}}) = 'win'
    if(nodes[0].mu_hat >= theta){
      for(int i=N_size-1; i>=0; i--){
        Tree_Alg::Node &node = nodes[i];
        if(node.is_leaf == true){
          if(node.mu_hat >= theta){
            node.Z = node.N * KLdiv(node.mu_hat, theta);
          }
          else{
            node.Z = 0;
          }
        }
        else if(node.type == MAXNODE){
          node.Z = 0;
          for(int c: node.child){
            node.Z += nodes[c].Z;
          }
        }
        else if(node.type == MINNODE){
          node.Z = 1e100;
          for(int c: node.child){
            node.Z = std::min(node.Z, nodes[c].Z);
          }
        }
      }
    }
    // a_{s_0}(\hat{\bm{\mu}}) = 'lose'
    else{
      for(int i=N_size-1; i>=0; i--){
        Tree_Alg::Node &node = nodes[i];
        if(node.is_leaf == true){
          if(node.mu_hat < theta){
            node.Z = node.N * KLdiv(node.mu_hat, theta);
          }
          else{
            node.Z = 0;
          }
        }
        else if(node.type == MINNODE){
          node.Z = 0;
          for(int c: node.child){
            node.Z += nodes[c].Z;
          }
        }
        else if(node.type == MAXNODE){
          node.Z = 1e100;
          for(int c: node.child){
            node.Z = std::min(node.Z, nodes[c].Z);
          }
        }
      }
    }
  }

  void load_prob(Problem &prob);
};

class Tree_Prob{
  public:
  class Node{
    public:
    int id;
    int depth;
    bool is_leaf;
    NodeType type;
    int parent;
    std::vector<int> child;

    double true_mu;
    // int true_d_value;

    void print(){
      std::cout << "node_id: " << this->id << "\n";
      std::cout << "depth: " << this->depth << "\n";
      std::cout << "is_leaf: " << this->is_leaf << "\n";
      std::cout << "type: " << this->type << "\n";
      std::cout << "parent: " << this->parent << "\n";
      std::cout << "child: " << this->child << "\n";
      std::cout << "true_mu: " << this->true_mu << "\n";
      std::cout << std::endl;
    }
  };

  // int N_size;
  // int L_size;
  std::vector<Node> nodes;

  std::vector<int> l_to_n; // l_to_n[leaf_id] -> 対応するnode_id
  std::vector<int> n_to_l; // n_to_l[node_id] -> 対応するleaf_id

  inline Node &operator[](int i){
    return this->nodes[i];
  }
  inline const Node &operator[](int i) const{
    return this->nodes[i];
  }

  int get_leaf_id(int node_id){
    return n_to_l[node_id];
  }
  int get_node_id(int leaf_id){
    return l_to_n[leaf_id];
  }

  void print() const{
    // std::cout << "N_size: " << this->N_size << "\n";
    // std::cout << "L_size: " << this->L_size << "\n";
    std::cout << "l_to_n: " << this->l_to_n << "\n";
    std::cout << "n_to_l: " << this->n_to_l << "\n";
    std::cout << std::endl;
  }
};


class Problem{
  public:
  int N_size; // 木のサイズ
  int L_size;
  Tree_Prob tree;

  double theta; // しきい値
  // double delta; // 失敗率

  bool is_win; // 真の答え


  void read_probfile(const std::string filename){
    std::ifstream ifs(filename, std::ios::in);
    if(!ifs){
      std::cerr << "error! Problem::read_probfile: file_open_failed" << std::endl;
      exit(1);
    }
    std::string tmp;

    ifs >> tmp; // [size]
    ifs >> this->N_size;
    this->tree.nodes = std::vector<Tree_Prob::Node>(this->N_size);

    ifs >> tmp; // [answer]
    ifs >> this->theta;
    ifs >> tmp;
    if(tmp == "win"){
      this->is_win = true;
    }
    else{
      this->is_win = false;
    }

    ifs >> tmp; // [nodes]
    std::list<int> l_to_n_tmp = {};
    this->tree.n_to_l = std::vector<int>(this->N_size);
    int leaf_count = 0;

    for(int id=0; id<this->N_size; id++){
      ifs >> this->tree[id].id;
      ifs >> this->tree[id].depth;
      ifs >> tmp;
      if(tmp == "true"){
        this->tree[id].is_leaf = true;
        tree.n_to_l[id] = leaf_count;
        leaf_count++;
        l_to_n_tmp.push_back(id);
      }
      else{
        this->tree[id].is_leaf = false;
        tree.n_to_l[id] = -1;
      }
      ifs >> tmp;
      if(tmp == "MAX") this->tree[id].type = MAXNODE;
      else this->tree[id].type = MINNODE;
      ifs >> this->tree[id].true_mu;
      ifs >> this->tree[id].parent;
      int child_num;
      ifs >> child_num;
      this->tree[id].child = std::vector<int>(child_num);
      for(int c=0; c<child_num; c++){
        ifs >> this->tree[id].child[c];
      }
    }
    this->L_size = leaf_count;
    this->tree.l_to_n = std::vector<int>(l_to_n_tmp.begin(), l_to_n_tmp.end());

    ifs >> tmp;
    if(tmp != "end"){
      std::cout << "error! read_probfile() : couldn't find \"end\"." << std::endl;
      exit(1);
    }
    ifs.close();
  }

  void print() const {
    std::cout << "N_size: " << this->N_size << "\n";
    std::cout << "L_size: " << this->L_size << "\n";
    std::cout << "theta: " << this->theta << "\n";
    this->tree.print();
    std::cout << std::endl;
  }
};


void Tree_Alg::load_prob(Problem &prob){
  this->nodes = std::vector<Tree_Alg::Node>(prob.N_size);
  this->l_to_n = prob.tree.l_to_n;
  this->n_to_l = prob.tree.n_to_l;

  for(int i=0; i<prob.N_size; i++){
    this->nodes[i].id = prob.tree[i].id;
    this->nodes[i].depth = prob.tree[i].depth;
    this->nodes[i].is_leaf = prob.tree[i].is_leaf;
    this->nodes[i].type = prob.tree[i].type;
    this->nodes[i].parent = prob.tree[i].parent;
    this->nodes[i].child = prob.tree[i].child;
  }

  for(int i=0; i<prob.N_size; i++){
    this->nodes[i].N = 0;
    this->nodes[i].reward_sum = 0;
    this->nodes[i].mu_hat = prob.tree[i].true_mu;
    this->nodes[i].d_value = 0;
    this->nodes[i].sub_weight = std::vector<double>(prob.L_size);
  }
}


// 報酬生成 兼 選択回数や割合の統計
class Oracle{
  private:
  int L_size;
  std::vector<double> true_mu;
  std::vector<int> N;
  int N_total;

  Experiment exp;
  Algorithm_base *alg;

  // 各記録step時点における選択比率の誤差を保存する
  std::vector<double> optimal_weight;
  std::vector<double> weight_diff; // ある時刻におけるweightと真のweightとの差
  std::vector<double> sample_diff; // ある時刻における選択比率と真のweightとの差
  int log_num = T_END / T_STEP;

  public:
  Oracle(Problem &prob, Experiment exp, Algorithm_base* alg){
    this->L_size = prob.L_size;
    this->true_mu = std::vector<double>(L_size);
    this->N = std::vector<int>(L_size, 0);
    for(int i=0; i<L_size; i++){
      int node_id = prob.tree.get_node_id(i);
      this->true_mu[i] = prob.tree[node_id].true_mu;
    }
    this->N_total = 0;

    this->exp = exp;
    this->alg = alg;

    if(exp == Experiment::SAMPLING_PROPORTION){
      this->weight_diff = std::vector<double>(log_num);
      this->sample_diff = std::vector<double>(log_num);
    }
  }

  void set_optimal_weight(std::vector<double> &weight){
    this->optimal_weight = weight;
  }

  int draw(int leaf_id){
    N[leaf_id]++;
    N_total++;
    if(exp == Experiment::SAMPLING_PROPORTION && (N_total % T_STEP == 0)) record_diff();

    return bernoulli(this->true_mu[leaf_id]);
  }

  void record_diff();

  void print(){
    std::cout << "L_size: " << this->L_size << "\n";
    std::cout << "true_mu: " << this->true_mu << "\n";
    std::cout << "N: " << this->N << "\n";
    std::cout << "N_total: " << this->N_total << "\n";
    std::cout << std::endl;
  }

  int get_N_total(){
    return N_total;
  }

  std::vector<double> &get_weight_diff(){
    return this->weight_diff;
  }
  std::vector<double> &get_sample_diff(){
    return this->sample_diff;
  }

  void reset_N(){
    for(int i=0; i<L_size; i++){
      this->N[i] = 0;
    }
    this->N_total = 0;
  }

};


// d次元で、要素の和が1になる非負のランダムなベクトルを返す
std::vector<double> random_weight(int d){
  std::gamma_distribution<double> gamma(1.0, 1.0);

  std::vector<double> ret(d);

  double sum_y = 0;
	for (int i = 0; i < d; i++) {
		double y = gamma(mt);
		sum_y += y;
		ret[i] = y;
	}
	std::for_each(ret.begin(), ret.end(), [sum_y](double &v) { v /= sum_y; });
  return ret;
}


void calc_weight(Tree_Alg &tree, double theta){
  int N_size = tree.nodes.size();
  int L_size = tree.nodes[0].sub_weight.size();

  // a_{s_0}(\hat{\bm{\mu}}) = 'win'
  if(tree[0].mu_hat >= theta){
    for(int i=N_size-1; i>=0; i--){
      Tree_Alg::Node &node = tree[i];
      // 葉ノードのとき
      if(node.is_leaf == true){
        for(int ell=0; ell<L_size; ell++){
          node.sub_weight[ell] = 0;
        }
        node.sub_weight[tree.n_to_l[i]] = 1;
      }
      // d > 0のとき
      else if(node.d_value > 0){
        // MAXノード
        if(node.type == MAXNODE){
          int best_child = 0;
          double best_d = tree[node.child[0]].d_value;
          for(int c=1; c<node.child.size(); c++){
            if(best_d < tree[node.child[c]].d_value){
              best_d = tree[node.child[c]].d_value;
              best_child = c;
            }
          }
          node.sub_weight = tree[node.child[best_child]].sub_weight;
        }
        // MINノード
        else{
          for(int ell=0; ell<L_size; ell++){
            node.sub_weight[ell] = 0;
          }
          double sum_d_inv = 0;
          for(int c=0; c<node.child.size(); c++){
            sum_d_inv += 1.0 / tree[node.child[c]].d_value;
          }
          for(int c=0; c<node.child.size(); c++){
            for(int ell=0; ell<L_size; ell++){
              node.sub_weight[ell] += tree[node.child[c]].sub_weight[ell] * (1.0 / tree[node.child[c]].d_value) / sum_d_inv;
            }
          }
        }
      }
      // d = 0のとき : 任意の重み
      else{
        if(weight_otherwise == WeightType::UNIFORM){
          for(int ell=0; ell<L_size; ell++){
            node.sub_weight[ell] = 1.0 / L_size; 
            // とりあえず一様とする
            // 本当は部分木内での重みを計算するが、どうせroot以外ではここの値を使わないので、全葉ノードに1/L_sizeであてておく
          }
        }
        else if(weight_otherwise == WeightType::RANDOM){
          auto weight = random_weight(L_size);
          for(int ell=0; ell<L_size; ell++){
            node.sub_weight[ell] = weight[ell];
            // とりあえず一様とする
            // 本当は部分木内での重みを計算するが、どうせroot以外ではここの値を使わないので、全葉ノードに1/L_sizeであてておく
          }
        }
        // node.sub_weight[0] = 1;
      }
    }
  }
  // a_{s_0}(\hat{\bm{\mu}}) = 'lose'
  else{
    for(int i=N_size-1; i>=0; i--){
      Tree_Alg::Node &node = tree[i];
      // 葉ノードのとき
      if(node.is_leaf == true){
        for(int ell=0; ell<L_size; ell++){
          node.sub_weight[ell] = 0;
        }
        node.sub_weight[tree.n_to_l[i]] = 1;
      }
      // d > 0のとき
      else if(node.d_value > 0){
        // MINノード
        if(node.type == MINNODE){
          int best_child = 0;
          double best_d = tree[node.child[0]].d_value;
          for(int c=1; c<node.child.size(); c++){
            if(best_d < tree[node.child[c]].d_value){
              best_d = tree[node.child[c]].d_value;
              best_child = c;
            }
          }
          node.sub_weight = tree[node.child[best_child]].sub_weight;
        }
        // MAXノード
        else{
          for(int ell=0; ell<L_size; ell++){
            node.sub_weight[ell] = 0;
          }
          double sum_d_inv = 0;
          for(int c=0; c<node.child.size(); c++){
            sum_d_inv += 1.0 / tree[node.child[c]].d_value;
          }
          for(int c=0; c<node.child.size(); c++){
            for(int ell=0; ell<L_size; ell++){
              node.sub_weight[ell] += tree[node.child[c]].sub_weight[ell] * (1.0 / tree[node.child[c]].d_value) / sum_d_inv;
            }
          }
        }
      }
      // d = 0のとき : 任意の重み
      else{
        if(weight_otherwise == WeightType::UNIFORM){
          for(int ell=0; ell<L_size; ell++){
            node.sub_weight[ell] = 1.0 / L_size; 
            // とりあえず一様とする
            // 本当は部分木内での重みを計算するが、どうせroot以外ではここの値を使わないので、全葉ノードに1/L_sizeであてておく
          }
        }
        else if(weight_otherwise == WeightType::RANDOM){
          auto weight = random_weight(L_size);
          for(int ell=0; ell<L_size; ell++){
            node.sub_weight[ell] = weight[ell];
            // 本当は部分木内での重みを計算するが、どうせroot以外ではここの値を使わないので、全葉ノードで計算
          }
        }
      }
    }
  }
}

class Algorithm_base{
  public:
  int N_size;
  int L_size;
  Tree_Alg tree;

  double theta;
  double delta_coef;
  double delta_exp;

  Experiment exp;

  virtual void set_param(const Problem &prob, const double delta_coef, const double delta_exp, const Experiment exp) = 0;
  virtual bool run(Oracle &oracle) = 0; // winならtrueを返す
  virtual void print() = 0;
  virtual ~Algorithm_base(){}
};

std::vector<int> debug_vec(20);

class Alg_Dtracking : public Algorithm_base{
  public:

  void set_param(const Problem &prob, const double delta_coef, const double delta_exp, const Experiment exp){  
    this->N_size = prob.N_size;
    this->L_size = prob.L_size;
    // this->tree.N_size = prob.tree.N_size;
    // this->tree.L_size = prob.tree.L_size;
    this->tree.nodes = std::vector<Tree_Alg::Node>(N_size);
    this->tree.l_to_n = prob.tree.l_to_n;
    this->tree.n_to_l = prob.tree.n_to_l;

    debug_vec = prob.tree.l_to_n;

    for(int i=0; i<N_size; i++){
      this->tree[i].id = prob.tree[i].id;
      this->tree[i].depth = prob.tree[i].depth;
      this->tree[i].is_leaf = prob.tree[i].is_leaf;
      this->tree[i].type = prob.tree[i].type;
      this->tree[i].parent = prob.tree[i].parent;
      this->tree[i].child = prob.tree[i].child;
    }
    this->theta = prob.theta;
    this->delta_coef = delta_coef;
    this->delta_exp = delta_exp;

    for(int i=0; i<this->N_size; i++){
      this->tree[i].N = 0;
      this->tree[i].reward_sum = 0;
      this->tree[i].mu_hat = 0;
      this->tree[i].d_value = 0;
      this->tree[i].sub_weight = std::vector<double>(this->L_size);
    }

    this->exp = exp;
  }

  void reset(){
    for(int i=0; i<this->N_size; i++){
      this->tree[i].N = 0;
      this->tree[i].reward_sum = 0;
      this->tree[i].mu_hat = 0;
      this->tree[i].d_value = 0;
      // this->tree[i].sub_weight = {};
    }
  }

  double beta(int t){ 
    // return std::log(std::log(t+1) / this->delta);
    return std::log(std::log(t+1)) - std::log(this->delta_coef) - this->delta_exp * std::log(10);
  }

  // true: 停止, false: 継続
  bool is_stop(int t){
    if(this->exp == Experiment::SAMPLING_PROPORTION){
      if(t >= T_END) return true;
      else return false;
    }

    // tree.calc_mu_hat();
    // tree.calc_Z(theta);

    double b = beta(t);
    if(tree[0].Z >= b || -tree[0].Z >= b){
      return true;
    }
    else{
      return false;
    }
  }


  // winならtrueを返す
  bool run(Oracle &oracle){
    bool debug1 = false;

    // initialize
    this->reset();

    // draw each arm once
    for(int ell=0; ell<L_size; ell++){
      double reward = oracle.draw(ell);
      tree.update_mu_hat(reward, ell);
    }

    int t = 1;
    while(true){
      std::vector<int> N(L_size);
      std::vector<double> mu_hat(L_size);
      for(int i=0; i<L_size; i++){
        int node_id = tree.get_node_id(i);
        N[i] = tree[node_id].N;
        mu_hat[i] = tree[node_id].mu_hat;
      }

      int next_leaf_id = -1;

      double threshold = std::sqrt(t) - L_size/2.0;
      int least_leaf = 0;
      int least_count = N[0];
      for(int i=1; i<L_size; i++){
        if(least_count > N[i]){
          least_count = N[i];
          least_leaf = i;
        }
      }

      if(least_count < threshold){
        next_leaf_id = least_leaf;
        debug1 = true;
      }
      else{
        debug1 = false;
        tree.calc_d_value(theta);
        calc_weight(tree, theta);
        std::vector<double> &weight = tree[0].sub_weight;

        next_leaf_id = -1;
        double max_diff = -1e100;
        for(int i=0; i<L_size; i++){
          double diff = t * weight[i] - N[i];
          // double diff = weight[i] / static_cast<double>(N[i]);
          if(diff > max_diff){
            max_diff = diff;
            next_leaf_id = i;
          }
        }
      }

      if(next_leaf_id < 0 || next_leaf_id >= L_size){
        breakfunc("illegal leaf_id!");
      }


      // Tree_Alg treelog = tree;

      // draw leaf
      double reward = oracle.draw(next_leaf_id);
      tree.update_mu_hat(reward, next_leaf_id);
      tree.calc_Z(theta);

      if(is_stop(t)){
        if(tree[0].mu_hat >= theta){
          return true;
        }
        else{
          return false;
        }
      }

      // breakfunc_if(t > 10000, "too many t!");

      t++;
    }


    return 0;
  }



  void print(){
    std::cout << "algorighm: D-track\n";
    std::cout << "N_size: " << this->N_size << "\n";
    std::cout << "L_size: " << this->N_size << "\n";
    std::cout << "theta: " << this->theta << "\n";
    std::cout << "delta: " << this->delta_coef << "e" << this->delta_exp << "\n";
    // std::cout << "nodes:\n"
    // for(int i=0; i<this->N_size; i++){
    //   this->tree[i].print();
    // }
    std::cout << std::endl;
  }
};

class Alg_Ctracking : public Algorithm_base{
  public:
  std::vector<double> weight_sum;

  void set_param(const Problem &prob, const double delta_coef, const double delta_exp, const Experiment exp){  
    this->N_size = prob.N_size;
    this->L_size = prob.L_size;
    // this->tree.N_size = prob.tree.N_size;
    // this->tree.L_size = prob.tree.L_size;
    this->tree.nodes = std::vector<Tree_Alg::Node>(N_size);
    this->tree.l_to_n = prob.tree.l_to_n;
    this->tree.n_to_l = prob.tree.n_to_l;

    debug_vec = prob.tree.l_to_n;

    for(int i=0; i<N_size; i++){
      this->tree[i].id = prob.tree[i].id;
      this->tree[i].depth = prob.tree[i].depth;
      this->tree[i].is_leaf = prob.tree[i].is_leaf;
      this->tree[i].type = prob.tree[i].type;
      this->tree[i].parent = prob.tree[i].parent;
      this->tree[i].child = prob.tree[i].child;
    }
    this->theta = prob.theta;
    this->delta_coef = delta_coef;
    this->delta_exp = delta_exp;

    for(int i=0; i<this->N_size; i++){
      this->tree[i].N = 0;
      this->tree[i].reward_sum = 0;
      this->tree[i].mu_hat = 0;
      this->tree[i].d_value = 0;
      this->tree[i].sub_weight = std::vector<double>(this->L_size);
    }

    this->exp = exp;

    this->weight_sum = std::vector<double>(this->L_size, 0);
  }

  void reset(){
    for(int i=0; i<this->N_size; i++){
      this->tree[i].N = 0;
      this->tree[i].reward_sum = 0;
      this->tree[i].mu_hat = 0;
      this->tree[i].d_value = 0;
      // this->tree[i].sub_weight = {};
    }
  }

  double beta(int t){ 
    // return std::log(std::log(t+1) / this->delta);
    return std::log(std::log(t+1)) - std::log(this->delta_coef) - this->delta_exp * std::log(10);
  }

  // true: 停止, false: 継続
  bool is_stop(int t){
    if(this->exp == Experiment::SAMPLING_PROPORTION){
      if(t >= T_END) return true;
      else return false;
    }

    // tree.calc_mu_hat();
    // tree.calc_Z(theta);

    double b = beta(t);
    if(tree[0].Z >= b || -tree[0].Z >= b){
      return true;
    }
    else{
      return false;
    }
  }


  // ChatGPTに書いてもらったけどちゃんと確認してません。チャッピーは賢いなあ
  std::vector<double> adjust_weights(const std::vector<double>& weight, double epsilon) {
    int n = (int)weight.size();
    const double EPS = 1e-12;   // 二分探索の精度

    // 与えられた d で条件を満たすベクトルが存在するか？
    auto feasible = [&](double d) -> bool {
        double min_sum = 0.0;
        double max_sum = 0.0;
        for (int i = 0; i < n; ++i) {
            double li = std::max({epsilon, weight[i] - d, 0.0});
            double ui = std::min(1.0, weight[i] + d);
            if (li > ui) return false;  // 区間が空
            min_sum += li;
            max_sum += ui;
        }
        return (min_sum <= 1.0 + 1e-15 && 1.0 <= max_sum + 1e-15);
    };

    // d を [0, 1] の範囲で二分探索
    double lo = 0.0;
    double hi = 1.0; // epsilon <= 1/n なので d=1 なら必ず可
    while (hi - lo > EPS) {
        double mid = 0.5 * (lo + hi);
        if (feasible(mid)) hi = mid;
        else lo = mid;
    }
    double d = hi;

    // d で実際の weight2 を構成
    std::vector<double> result(n);
    std::vector<double> li(n), ui(n);
    double min_sum = 0.0;
    for (int i = 0; i < n; ++i) {
        li[i] = std::max({epsilon, weight[i] - d, 0.0});
        ui[i] = std::min(1.0, weight[i] + d);
        if (li[i] > ui[i]) {
            // 数値誤差対策として、少し詰める
            li[i] = ui[i];
        }
        result[i] = li[i];
        min_sum += li[i];
    }

    double rem = 1.0 - min_sum; // まだ足りない分を各成分に配る
    for (int i = 0; i < n && rem > 0.0; ++i) {
        double can_add = ui[i] - result[i];
        double add = std::min(can_add, rem);
        result[i] += add;
        rem -= add;
    }

    // 数値誤差で和が少しズレている場合の微調整（任意）
    double sum_res = accumulate(result.begin(), result.end(), 0.0);
    if (fabs(sum_res - 1.0) > 1e-10) {
        for (int i = 0; i < n; ++i) result[i] /= sum_res;
    }

    return result;
  }


  // winならtrueを返す
  bool run(Oracle &oracle){
    bool debug1 = false;

    // initialize
    this->reset();

    // draw each arm once
    for(int ell=0; ell<L_size; ell++){
      double reward = oracle.draw(ell);
      tree.update_mu_hat(reward, ell);
    }

    int t = 1;
    while(true){
      std::vector<int> N(L_size);
      std::vector<double> mu_hat(L_size);
      for(int i=0; i<L_size; i++){
        int node_id = tree.get_node_id(i);
        N[i] = tree[node_id].N;
        mu_hat[i] = tree[node_id].mu_hat;
      }

      int next_leaf_id = -1;
      tree.calc_d_value(theta);
      calc_weight(tree, theta);
      std::vector<double> weight = tree[0].sub_weight;
      double epsilon = 1.0 / (2 * std::sqrt(L_size * L_size + t));
      weight = adjust_weights(weight, epsilon);
      for(int ell=0; ell<L_size; ell++){
        this->weight_sum[ell] += weight[ell];
      }

      // なんか誤差が出るのでちょっと無理やり補正...
      double sum = 0;
      for(int ell=0; ell<L_size; ell++){
        sum += weight_sum[ell];
      }
      for(int ell=0; ell<L_size; ell++){
        weight_sum[ell] = weight_sum[ell] * (static_cast<double>(t) / static_cast<double>(sum));
      }


      next_leaf_id = -1;
      double max_diff = -1e100;
      for(int i=0; i<L_size; i++){
        double diff = this->weight_sum[i] - N[i];
        if(diff > max_diff){
          max_diff = diff;
          next_leaf_id = i;
        }
      }

      if(next_leaf_id < 0 || next_leaf_id >= L_size){
        breakfunc("illegal leaf_id!");
      }

      // Tree_Alg treelog = tree;

      // draw leaf
      double reward = oracle.draw(next_leaf_id);
      tree.update_mu_hat(reward, next_leaf_id);
      tree.calc_Z(theta);

      if(is_stop(t)){
        if(tree[0].mu_hat >= theta){
          return true;
        }
        else{
          return false;
        }
      }

      // breakfunc_if(t > 10000, "too many t!");

      t++;
    }


    return 0;
  }



  void print(){
    std::cout << "algorighm: C-track\n";
    std::cout << "N_size: " << this->N_size << "\n";
    std::cout << "L_size: " << this->N_size << "\n";
    std::cout << "theta: " << this->theta << "\n";
    std::cout << "delta: " << this->delta_coef << "e" << this->delta_exp << "\n";
    // std::cout << "nodes:\n"
    // for(int i=0; i<this->N_size; i++){
    //   this->tree[i].print();
    // }
    std::cout << std::endl;
  }
};

// w/N型D-Tracking
class Alg_WN_Dtracking : public Algorithm_base{
  public:

  void set_param(const Problem &prob, const double delta_coef, const double delta_exp, const Experiment exp){  
    this->N_size = prob.N_size;
    this->L_size = prob.L_size;
    // this->tree.N_size = prob.tree.N_size;
    // this->tree.L_size = prob.tree.L_size;
    this->tree.nodes = std::vector<Tree_Alg::Node>(N_size);
    this->tree.l_to_n = prob.tree.l_to_n;
    this->tree.n_to_l = prob.tree.n_to_l;

    debug_vec = prob.tree.l_to_n;

    for(int i=0; i<N_size; i++){
      this->tree[i].id = prob.tree[i].id;
      this->tree[i].depth = prob.tree[i].depth;
      this->tree[i].is_leaf = prob.tree[i].is_leaf;
      this->tree[i].type = prob.tree[i].type;
      this->tree[i].parent = prob.tree[i].parent;
      this->tree[i].child = prob.tree[i].child;
    }
    this->theta = prob.theta;
    this->delta_coef = delta_coef;
    this->delta_exp = delta_exp;

    for(int i=0; i<this->N_size; i++){
      this->tree[i].N = 0;
      this->tree[i].reward_sum = 0;
      this->tree[i].mu_hat = 0;
      this->tree[i].d_value = 0;
      this->tree[i].sub_weight = std::vector<double>(this->L_size);
    }

    this->exp = exp;
  }

  void reset(){
    for(int i=0; i<this->N_size; i++){
      this->tree[i].N = 0;
      this->tree[i].reward_sum = 0;
      this->tree[i].mu_hat = 0;
      this->tree[i].d_value = 0;
      // this->tree[i].sub_weight = {};
    }
  }

  double beta(int t){ 
    // return std::log(std::log(t+1) / this->delta);
    return std::log(std::log(t+1)) - std::log(this->delta_coef) - this->delta_exp * std::log(10);
  }

  // true: 停止, false: 継続
  bool is_stop(int t){
    if(this->exp == Experiment::SAMPLING_PROPORTION){
      if(t >= T_END) return true;
      else return false;
    }

    // tree.calc_mu_hat();
    // tree.calc_Z(theta);

    double b = beta(t);
    if(tree[0].Z >= b || -tree[0].Z >= b){
      return true;
    }
    else{
      return false;
    }
  }


  // winならtrueを返す
  bool run(Oracle &oracle){
    bool debug1 = false;

    // initialize
    this->reset();

    // draw each arm once
    for(int ell=0; ell<L_size; ell++){
      double reward = oracle.draw(ell);
      tree.update_mu_hat(reward, ell);
    }

    int t = 1;
    while(true){
      std::vector<int> N(L_size);
      std::vector<double> mu_hat(L_size);
      for(int i=0; i<L_size; i++){
        int node_id = tree.get_node_id(i);
        N[i] = tree[node_id].N;
        mu_hat[i] = tree[node_id].mu_hat;
      }

      int next_leaf_id = -1;

      double threshold = std::sqrt(t) - L_size/2.0;
      int least_leaf = 0;
      int least_count = N[0];
      for(int i=1; i<L_size; i++){
        if(least_count > N[i]){
          least_count = N[i];
          least_leaf = i;
        }
      }

      if(least_count < threshold){
        next_leaf_id = least_leaf;
        debug1 = true;
      }
      else{
        debug1 = false;
        tree.calc_d_value(theta);
        calc_weight(tree, theta);
        std::vector<double> &weight = tree[0].sub_weight;

        next_leaf_id = -1;
        double max_diff = -1e100;
        for(int i=0; i<L_size; i++){
          // double diff = t * weight[i] - N[i];
          double diff = weight[i] / static_cast<double>(N[i]);
          if(diff > max_diff){
            max_diff = diff;
            next_leaf_id = i;
          }
        }
      }

      if(next_leaf_id < 0 || next_leaf_id >= L_size){
        breakfunc("illegal leaf_id!");
      }


      // Tree_Alg treelog = tree;

      // draw leaf
      double reward = oracle.draw(next_leaf_id);
      tree.update_mu_hat(reward, next_leaf_id);
      tree.calc_Z(theta);

      if(is_stop(t)){
        if(tree[0].mu_hat >= theta){
          return true;
        }
        else{
          return false;
        }
      }

      // breakfunc_if(t > 10000, "too many t!");

      t++;
    }


    return 0;
  }



  void print(){
    std::cout << "algorighm: D-track\n";
    std::cout << "N_size: " << this->N_size << "\n";
    std::cout << "L_size: " << this->N_size << "\n";
    std::cout << "theta: " << this->theta << "\n";
    std::cout << "delta: " << this->delta_coef << "e" << this->delta_exp << "\n";
    // std::cout << "nodes:\n"
    // for(int i=0; i<this->N_size; i++){
    //   this->tree[i].print();
    // }
    std::cout << std::endl;
  }
};



class Alg_RandomTracking : public Algorithm_base{
  public:

  void set_param(const Problem &prob, const double delta_coef, const double delta_exp, const Experiment exp){  
    this->N_size = prob.N_size;
    this->L_size = prob.L_size;
    // this->tree.N_size = prob.tree.N_size;
    // this->tree.L_size = prob.tree.L_size;
    this->tree.nodes = std::vector<Tree_Alg::Node>(N_size);
    this->tree.l_to_n = prob.tree.l_to_n;
    this->tree.n_to_l = prob.tree.n_to_l;

    debug_vec = prob.tree.l_to_n;

    for(int i=0; i<N_size; i++){
      this->tree[i].id = prob.tree[i].id;
      this->tree[i].depth = prob.tree[i].depth;
      this->tree[i].is_leaf = prob.tree[i].is_leaf;
      this->tree[i].type = prob.tree[i].type;
      this->tree[i].parent = prob.tree[i].parent;
      this->tree[i].child = prob.tree[i].child;
    }
    this->theta = prob.theta;
    this->delta_coef = delta_coef;
    this->delta_exp = delta_exp;

    for(int i=0; i<this->N_size; i++){
      this->tree[i].N = 0;
      this->tree[i].reward_sum = 0;
      this->tree[i].mu_hat = 0;
      this->tree[i].d_value = 0;
      this->tree[i].sub_weight = std::vector<double>(this->L_size);
    }

    this->exp = exp;
  }

  void reset(){
    for(int i=0; i<this->N_size; i++){
      this->tree[i].N = 0;
      this->tree[i].reward_sum = 0;
      this->tree[i].mu_hat = 0;
      this->tree[i].d_value = 0;
      // this->tree[i].sub_weight = {};
    }
  }

  double beta(int t){ 
    // return std::log(std::log(t+1) / this->delta);
    return std::log(std::log(t+1)) - std::log(this->delta_coef) - this->delta_exp * std::log(10);
  }

  // true: 停止, false: 継続
  bool is_stop(int t){
    if(this->exp == Experiment::SAMPLING_PROPORTION){
      if(t >= T_END) return true;
      else return false;
    }

    // tree.calc_mu_hat();

    double b = beta(t);
    if(tree[0].Z >= b || -tree[0].Z >= b){
      return true;
    }
    else{
      return false;
    }
  }


  // winならtrueを返す
  bool run(Oracle &oracle){
    // initialize
    this->reset();

    // draw each arm once
    for(int ell=0; ell<L_size; ell++){
      double reward = oracle.draw(ell);
      tree.update_mu_hat(reward, ell);
    }

    int t = 1;
    while(true){

      std::vector<int> N(L_size);
      std::vector<double> mu_hat(L_size);
      for(int i=0; i<L_size; i++){
        int node_id = tree.get_node_id(i);
        N[i] = tree[node_id].N;
        mu_hat[i] = tree[node_id].mu_hat;
      }

      int next_leaf_id = -1;

      double threshold = std::sqrt(t) - L_size/2.0;
      int least_leaf = 0;
      int least_count = N[0];
      for(int i=1; i<L_size; i++){
        if(least_count > N[i]){
          least_count = N[i];
          least_leaf = i;
        }
      }

      if(least_count < threshold && !disable_forced_exploration){
      // if(least_count < threshold){
        next_leaf_id = least_leaf;
      }
      else{
        tree.calc_d_value(theta);
        calc_weight(tree, theta);
        std::vector<double> &weight = tree[0].sub_weight;

        next_leaf_id = L_size - 1;
        // Random-Tracking
        std::uniform_real_distribution<> rand(0.0,1.0);
        double r = rand(mt);
        for(int ell = 0; ell < L_size; ell++){
          if(r < weight[ell]){
            next_leaf_id = ell;
            break;
          }
          else{
            r -= weight[ell];
          }
        }
      }

      if(next_leaf_id < 0 || next_leaf_id >= L_size){
        breakfunc("illegal leaf_id!");
      }

      // draw leaf
      double reward = oracle.draw(next_leaf_id);
      tree.update_mu_hat(reward, next_leaf_id);
      tree.calc_Z(theta);

      if(is_stop(t)){
        if(tree[0].mu_hat >= theta){
          // if(t > 1000){
          //   breakfunc();
          // }
          return true;
        }
        else{
          return false;
        }
      }

      t++;
    }

    return 0;
  }

  void print(){
    std::cout << "algorighm: Random-Tracking\n";
    std::cout << "N_size: " << this->N_size << "\n";
    std::cout << "L_size: " << this->N_size << "\n";
    std::cout << "theta: " << this->theta << "\n";
    std::cout << "delta: " << this->delta_coef << "e" << this->delta_exp << "\n";
    std::cout << std::endl;
  }
};



class Alg_Ptracking : public Algorithm_base{
  public:
  std::vector<int> beta_a;
  std::vector<int> beta_b;

  Tree_Alg po_tree;

  void set_param(const Problem &prob, const double delta_coef, const double delta_exp, const Experiment exp){
    this->N_size = prob.N_size;
    this->L_size = prob.L_size;
    // this->tree.N_size = prob.tree.N_size;
    // this->tree.L_size = prob.tree.L_size;
    this->tree.nodes = std::vector<Tree_Alg::Node>(N_size);
    this->tree.l_to_n = prob.tree.l_to_n;
    this->tree.n_to_l = prob.tree.n_to_l;

    // debug_vec = prob.tree.l_to_n;

    this->beta_a = std::vector<int>(L_size,1);
    this->beta_b = std::vector<int>(L_size,1);
    this->po_tree.nodes = std::vector<Tree_Alg::Node>(N_size);
    this->po_tree.l_to_n = prob.tree.l_to_n;
    this->po_tree.n_to_l = prob.tree.n_to_l;

    for(int i=0; i<N_size; i++){
      this->tree[i].id = prob.tree[i].id;
      this->tree[i].depth = prob.tree[i].depth;
      this->tree[i].is_leaf = prob.tree[i].is_leaf;
      this->tree[i].type = prob.tree[i].type;
      this->tree[i].parent = prob.tree[i].parent;
      this->tree[i].child = prob.tree[i].child;

      this->po_tree[i].id = prob.tree[i].id;
      this->po_tree[i].depth = prob.tree[i].depth;
      this->po_tree[i].is_leaf = prob.tree[i].is_leaf;
      this->po_tree[i].type = prob.tree[i].type;
      this->po_tree[i].parent = prob.tree[i].parent;
      this->po_tree[i].child = prob.tree[i].child;
    }
    this->theta = prob.theta;
    this->delta_coef = delta_coef;
    this->delta_exp = delta_exp;

    for(int i=0; i<this->N_size; i++){
      this->tree[i].N = 0;
      this->tree[i].reward_sum = 0;
      this->tree[i].mu_hat = 0;
      this->tree[i].d_value = 0;
      this->tree[i].sub_weight = std::vector<double>(this->L_size);

      this->po_tree[i].N = 0;
      this->po_tree[i].reward_sum = 0;
      this->po_tree[i].mu_hat = 0;
      this->po_tree[i].d_value = 0;
      this->po_tree[i].sub_weight = std::vector<double>(this->L_size);
    }

    this->exp = exp;
  }

  void reset(){
    for(int i=0; i<this->N_size; i++){
      this->tree[i].N = 0;
      this->tree[i].reward_sum = 0;
      this->tree[i].mu_hat = 0;
      this->tree[i].d_value = 0;
      // this->tree[i].sub_weight = {};

      this->po_tree[i].N = 0;
      this->po_tree[i].reward_sum = 0;
      this->po_tree[i].mu_hat = 0;
      this->po_tree[i].d_value = 0;
    }
    for(int i=0; i<this->L_size; i++){
      this->beta_a[i] = 1;
      this->beta_b[i] = 1;
    }
  }

  double beta(int t){ 
    // return std::log(std::log(t+1) / this->delta);
    return std::log(std::log(t+1)) - std::log(this->delta_coef) - this->delta_exp * std::log(10);
  }

  // true: 停止, false: 継続
  bool is_stop(int t){
    if(this->exp == Experiment::SAMPLING_PROPORTION){
      if(t >= T_END) return true;
      else return false;
    }

    double b = beta(t);
    if(tree[0].Z >= b || -tree[0].Z >= b){
      return true;
    }
    else{
      return false;
    }
  }

  bool run(Oracle &oracle){
    // initialize
    this->reset();

    int t = 1;
    while(true){
      std::vector<double> po_param(L_size);
      for(int i=0; i<L_size; i++){
        boost::random::beta_distribution<> sample_beta(beta_a[i], beta_b[i]);
        po_param[i] = sample_beta(mt);

        po_tree[tree.l_to_n[i]].mu_hat = po_param[i];
        po_tree[tree.l_to_n[i]].N = tree[tree.l_to_n[i]].N;
      }
      for(int i=N_size-1; i>=0; i--){
        if(!po_tree[i].is_leaf){
          po_tree[i].N = 0;
          po_tree[i].mu_hat = 0;
          for(int c : po_tree[i].child){
            po_tree[i].N += po_tree[c].N;
            po_tree[i].mu_hat += po_tree[c].mu_hat;
          }
          po_tree[i].mu_hat / po_tree[i].N;
        }
      }
      po_tree.calc_mu_hat();
      po_tree.calc_d_value(theta);
      calc_weight(po_tree, theta);
      std::vector<double> &weight = po_tree[0].sub_weight;
      this->tree[0].sub_weight = po_tree[0].sub_weight;

      double best_val = -1;
      int next_leaf_id = -1;
      for(int i=0; i<L_size; i++){
        int N = tree[tree.l_to_n[i]].N;
        if(weight[i] == 0) continue; 
        else if(N == 0){
          best_val = 1e100; // large number
          next_leaf_id = i;
        }
        else{
          double val = weight[i] / N;
          if(val > best_val){
            best_val = val;
            next_leaf_id = i;
          }
        }
      }

      // draw leaf
      int reward = oracle.draw(next_leaf_id);
      Tree_Alg::Node &node = tree[tree.l_to_n[next_leaf_id]];
      node.N++;
      node.reward_sum += reward;
      node.mu_hat = node.reward_sum / static_cast<double>(node.N);
      tree.calc_mu_hat();
      tree.calc_Z(theta);
      
      beta_a[next_leaf_id] += reward;
      beta_b[next_leaf_id] += (1 - reward);

      std::vector<int> N(L_size);
      for(int i=0; i<L_size; i++){
        N[i] = tree[tree.l_to_n[i]].N;
      }
      
      if(is_stop(t)){
        if(tree[0].mu_hat >= theta){
          // std::cout << this->tree[0].sub_weight << std::endl;
          return true;
        }
        else{
          return false;
        }
      }
      

      t++;
    }

    // breakfunc("error!");
    return false;
  }



  void print(){
    std::cout << "algorighm: P-track\n";
    std::cout << "N_size: " << this->N_size << "\n";
    std::cout << "L_size: " << this->N_size << "\n";
    std::cout << "theta: " << this->theta << "\n";
    std::cout << "delta: " << this->delta_coef << "e" << this->delta_exp << "\n";
    // std::cout << "nodes:\n"
    // for(int i=0; i<this->N_size; i++){
    //   this->tree[i].print();
    // }
    std::cout << std::endl;
  }

  void print_mu_hat(){
    // std::vector<double> mu_hat(L_size);
    // for(int i=0; i<L_size; i++){
    //   mu_hat[i] = tree[tree.l_to_n[i]].mu_hat;
    // }
    std::vector<double> mu_hat(N_size);
    for(int i=0; i<N_size; i++){
      mu_hat[i] = tree[i].mu_hat;
    }
    std::cout << "mu_hat: " << mu_hat << "\n";
    std::cout << std::endl;
  }

  void print_nodes(){
    for(int i=0; i<N_size; i++){
      tree[i].print();
    }
  }
};



class Alg_UGapE : public Algorithm_base{
  public:
  double epsilon;
  std::vector<double> UCB;
  std::vector<double> LCB;
  std::vector<int> repr_child;
  std::vector<int> repr_leaf;

  void set_param(const Problem &prob, const double delta_coef, const double delta_exp, const Experiment exp){
    this->N_size = prob.N_size;
    this->L_size = prob.L_size;
    this->tree.nodes = std::vector<Tree_Alg::Node>(N_size);
    this->tree.l_to_n = prob.tree.l_to_n;
    this->tree.n_to_l = prob.tree.n_to_l;


    for(int i=0; i<N_size; i++){
      this->tree[i].id = prob.tree[i].id;
      this->tree[i].depth = prob.tree[i].depth;
      this->tree[i].is_leaf = prob.tree[i].is_leaf;
      this->tree[i].type = prob.tree[i].type;
      this->tree[i].parent = prob.tree[i].parent;
      this->tree[i].child = prob.tree[i].child;
    }
    this->theta = prob.theta;
    this->delta_coef = delta_coef;
    this->delta_exp = delta_exp;

    for(int i=0; i<this->N_size; i++){
      this->tree[i].N = 0;
      this->tree[i].reward_sum = 0;
      this->tree[i].mu_hat = 0;
      this->tree[i].d_value = 0;
    }

    this->UCB = std::vector<double>(N_size,1.0);
    this->LCB = std::vector<double>(N_size,0.0);
    this->repr_child = std::vector<int>(N_size,-1);
    this->repr_leaf = std::vector<int>(N_size,-1);

    this->exp = exp;

    this->epsilon = 0.0;
  }

  void reset(){
    for(int i=0; i<this->N_size; i++){
      this->tree[i].N = 0;
      this->tree[i].reward_sum = 0;
      this->tree[i].mu_hat = 0;
      this->tree[i].d_value = 0;
      // this->tree[i].sub_weight = {};

      this->UCB[i] = 1e50;
      this->LCB[i] = -1e50;
      this->repr_child[i] = -1;
      this->repr_leaf[i] = -1;

    }
    calc_LUCB();
    // this->next_index = 0;
  }

  double beta(int t){ 
    // return std::log(std::log(t+1) / this->delta);
    return std::log(std::log(t+1)) - std::log(this->delta_coef) - this->delta_exp * std::log(10);
  }

  void calc_LUCB(){
    for(int i=N_size-1; i>=0; i--){
      if(tree[i].is_leaf == true){
        if(tree[i].N == 0){
          this->UCB[i] = 1e50;
          this->LCB[i] = -1e50;
          this->repr_leaf[i] = tree.get_leaf_id(i);
        }
        else{
          double width = std::sqrt(beta(tree[i].N) / (2*tree[i].N));
          this->UCB[i] = tree[i].mu_hat + width;
          this->LCB[i] = tree[i].mu_hat - width;
          this->repr_leaf[i] = tree.get_leaf_id(i);
        }
      }
      else if(tree[i].type == MAXNODE){
        double Umax = -1e50;
        double Lmax = -1e50;
        int argmaxU = -1;
        for(int c : tree[i].child){
          if(UCB[c] > Umax){
            argmaxU = c;
            Umax = UCB[c];
          }
          Lmax = std::max(Lmax, LCB[c]);
        }
        this->UCB[i] = Umax;
        this->LCB[i] = Lmax;
        this->repr_child[i] = argmaxU;
        this->repr_leaf[i] = this->repr_leaf[argmaxU];
      }
      else{ // MINNODE
        double Umin = 1e50;
        double Lmin = 1e50;
        int argminL = -1;
        for(int c : tree[i].child){
          Umin = std::min(Umin, UCB[c]);
          if(LCB[c] < Lmin){
            argminL = c;
            Lmin = LCB[c];
          }
        }
        this->UCB[i] = Umin;
        this->LCB[i] = Lmin;
        this->repr_child[i] = argminL;
        this->repr_leaf[i] = this->repr_leaf[argminL];
      }
    }
  }

  bool run(Oracle &oracle){
    // initialize
    this->reset();

    // draw each arm once
    // since UCB=infty and LCB=-infty when N=0
    // for(int ell=0; ell<L_size; ell++){
    //   double reward = oracle.draw(ell);
    //   int node_id = tree.get_node_id(ell);
    //   tree[node_id].N = 1;
    //   tree[node_id].reward_sum = reward;
    //   tree[node_id].mu_hat = reward;
    // }
    // tree.calc_mu_hat();
    // calc_LUCB();

    // int root_child_num = tree[0].child.size();
    std::vector<double> B(N_size);

    int t = 1;
    while(true){
      double maxU = -1e51;
      int argmaxU = -1;
      double second_maxU = -1e51;
      int second_argmaxU = -1;
      for(int c : tree[0].child){
        if(maxU < UCB[c]){
          second_argmaxU = argmaxU;
          second_maxU = maxU;
          argmaxU = c;
          maxU = UCB[c];
        }
        else if(second_maxU < UCB[c]){
          second_argmaxU = c;
          second_maxU = UCB[c];
        }
      }
      for(int c : tree[0].child){
        if(c == argmaxU){
          B[c] = second_maxU - LCB[c];
        }
        else{
          B[c] = maxU - LCB[c];
        }
      }
      double minB = 1e51;
      int best = -1;
      int challenger = -1;
      for(int c : tree[0].child){
        if(minB > B[c]){
          best = c;
          minB = B[c];
        }
      }
      if(best == argmaxU){
        challenger = second_argmaxU;
      }
      else{
        challenger = argmaxU;
      }

      if(UCB[challenger] - LCB[best] < epsilon){ // stop
        tree.calc_mu_hat();
        // debug info
        // std::vector<int> N(L_size);
        // for(int i=0; i<L_size; i++){
        //   N[i] = tree[tree.l_to_n[i]].N;
        // }
        // breakfunc();
        if(tree[best].mu_hat >= theta){
          return true;
        }
        else{
          return false;
        }
      }
      else{
        int chosen_leaf = -1;
        if(UCB[best] - LCB[best] > UCB[challenger] - LCB[challenger]){
          chosen_leaf = repr_leaf[best];
        }
        else{
          chosen_leaf = repr_leaf[challenger];
        }
        double reward = oracle.draw(chosen_leaf);

        int node_id = tree.get_node_id(chosen_leaf);
        tree[node_id].N += 1;
        tree[node_id].reward_sum += reward;
        tree[node_id].mu_hat = tree[node_id].reward_sum / static_cast<double>(tree[node_id].N);
        calc_LUCB();
      }

      // if(t % 2000 == 0){
        
      //   std::vector<int> N(L_size);
      //   for(int i=0; i<L_size; i++){
      //     N[i] = tree[tree.l_to_n[i]].N;
      //   }
      //   std::cout << N << std::endl;
      //   std::cout << UCB << std::endl;
      //   std::cout << LCB << std::endl;
      //   std::cout << "best: " << best << std::endl; 
      //   std::cout << "challenger: " << challenger << std::endl;

      //   breakfunc();

      // }
      // if(t > 10000) breakfunc();
      t++;
    }

    // breakfunc("error!");
    return false;
  }

  void print(){
    std::cout << "algorighm: P-track\n";
    std::cout << "N_size: " << this->N_size << "\n";
    std::cout << "L_size: " << this->N_size << "\n";
    std::cout << "theta: " << this->theta << "\n";
    std::cout << "delta: " << this->delta_coef << "e" << this->delta_exp << "\n";
    // std::cout << "nodes:\n"
    // for(int i=0; i<this->N_size; i++){
    //   this->tree[i].print();
    // }
    std::cout << std::endl;
  }
};



class Alg_LUCBmicro : public Algorithm_base{
  public:
  double epsilon;
  std::vector<double> UCB;
  std::vector<double> LCB;
  std::vector<int> repr_child;
  std::vector<int> repr_leaf;

  void set_param(const Problem &prob, const double delta_coef, const double delta_exp, const Experiment exp){
    this->N_size = prob.N_size;
    this->L_size = prob.L_size;
    this->tree.nodes = std::vector<Tree_Alg::Node>(N_size);
    this->tree.l_to_n = prob.tree.l_to_n;
    this->tree.n_to_l = prob.tree.n_to_l;


    for(int i=0; i<N_size; i++){
      this->tree[i].id = prob.tree[i].id;
      this->tree[i].depth = prob.tree[i].depth;
      this->tree[i].is_leaf = prob.tree[i].is_leaf;
      this->tree[i].type = prob.tree[i].type;
      this->tree[i].parent = prob.tree[i].parent;
      this->tree[i].child = prob.tree[i].child;
    }
    this->theta = prob.theta;
    this->delta_coef = delta_coef;
    this->delta_exp = delta_exp;

    for(int i=0; i<this->N_size; i++){
      this->tree[i].N = 0;
      this->tree[i].reward_sum = 0;
      this->tree[i].mu_hat = 0;
      this->tree[i].d_value = 0;
    }

    this->UCB = std::vector<double>(N_size,1.0);
    this->LCB = std::vector<double>(N_size,0.0);
    this->repr_child = std::vector<int>(N_size,-1);
    this->repr_leaf = std::vector<int>(N_size,-1);

    this->exp = exp;

    this->epsilon = 0.0;
  }

  void reset(){
    for(int i=0; i<this->N_size; i++){
      this->tree[i].N = 0;
      this->tree[i].reward_sum = 0;
      this->tree[i].mu_hat = 0;
      this->tree[i].d_value = 0;
      // this->tree[i].sub_weight = {};

      this->UCB[i] = 1e50;
      this->LCB[i] = -1e50;
      this->repr_child[i] = -1;
      this->repr_leaf[i] = -1;

    }
    calc_LUCB();
    // this->next_index = 0;
  }

  double beta(int t){ 
    // return std::log(std::log(t+1) / this->delta);
    return std::log(std::log(t+1)) - std::log(this->delta_coef) - this->delta_exp * std::log(10);

    // deltaが1/2Lされている
    // return std::log(std::log(t+1)) - std::log(this->delta_coef / (2*L_size)) - this->delta_exp * std::log(10);
  }

  // 葉の数L
  // 腕の数K
  // f : R^L \to R^K は葉の評価値から腕の評価値を算出する関数
  // f = (f_j)_{j = 1, \ldots, K} とおく。
  // 
  // B,Cの算出に使うf_j(L)は、「各葉ノードi評価値をL(i)として計算したときの、腕jの評価値」
  // つまり信頼区間の再帰的な計算自体はUGapEと変わらない。
  // (葉における信頼区間は定義が少し違う。
  //  UGapEは純粋に(平均+-幅)
  //  LUCB-microも(平均+-幅)だが、UCB/LCBは単調減少/増加に制限している。(平均+幅が前の時刻のUCBを超えていたら、更新せず前の時刻の値を採用))
  // 
  // D(j, u, v) = { i \in [L] : [f_j(u), f_j(v)] \subset [u_i, v_i]}
  // 
  // 今回はD(j) = D(j, L, U)を使う
  // [f_j(L), f_j(U)] \subset [L(i), U(i)]
  // 実は MAXノードでは子ノードのうちargmax UCBを、MINノードでは子ノードのうちargmax LCBを追いかけることで得られる葉ノードは、
  // このiを満たす。
  // 
  // というわけで、D(B)とD(C)の両方を引く
  // 


  void calc_LUCB(){
    for(int i=N_size-1; i>=0; i--){
      if(tree[i].is_leaf == true){
        if(tree[i].N == 0){
          this->UCB[i] = 1e50;
          this->LCB[i] = -1e50;
          this->repr_leaf[i] = tree.get_leaf_id(i);
        }
        else{
          double width = std::sqrt(beta(tree[i].N) / (2 * tree[i].N));
          // double width = std::sqrt(2 * beta(tree[i].N) / (tree[i].N));
          this->UCB[i] = std::min(UCB[i], tree[i].mu_hat + width);
          this->LCB[i] = std::max(LCB[i], tree[i].mu_hat - width);
          this->repr_leaf[i] = tree.get_leaf_id(i);
        }
      }
      else if(tree[i].type == MAXNODE){
        double Umax = -1e50;
        double Lmax = -1e50;
        int argmaxU = -1;
        for(int c : tree[i].child){
          if(UCB[c] > Umax){
            argmaxU = c;
            Umax = UCB[c];
          }
          Lmax = std::max(Lmax, LCB[c]);
        }
        this->UCB[i] = Umax;
        this->LCB[i] = Lmax;
        this->repr_child[i] = argmaxU;
        this->repr_leaf[i] = this->repr_leaf[argmaxU];
      }
      else{ // MINNODE
        double Umin = 1e50;
        double Lmin = 1e50;
        int argminL = -1;
        for(int c : tree[i].child){
          Umin = std::min(Umin, UCB[c]);
          if(LCB[c] < Lmin){
            argminL = c;
            Lmin = LCB[c];
          }
        }
        this->UCB[i] = Umin;
        this->LCB[i] = Lmin;
        this->repr_child[i] = argminL;
        this->repr_leaf[i] = this->repr_leaf[argminL];
      }
    }
  }

  bool run(Oracle &oracle){
    // initialize
    this->reset();

    std::vector<double> B(N_size);

    int t = 1;
    while(true){
      double maxL = -1e51;
      int argmaxL = -1;
      for(int c : tree[0].child){
        if(maxL < LCB[c]){
          argmaxL = c;
          maxL = LCB[c];
        }
      }

      double maxU = -1e51;
      double argmaxU = -1;
      for(int c : tree[0].child){
        if(maxU < UCB[c] && c != argmaxL){
          argmaxU = c;
          maxU = UCB[c];
        }
      }

      int I_leafID = repr_leaf[argmaxL];
      int I_nodeID = tree.get_node_id(I_leafID);
      int J_leafID = repr_leaf[argmaxU];
      int J_nodeID = tree.get_node_id(J_leafID);

      double reward_I = oracle.draw(I_leafID);
      double reward_J = oracle.draw(J_leafID);

      tree[I_nodeID].N += 1;
      tree[I_nodeID].reward_sum += reward_I;
      tree[I_nodeID].mu_hat = tree[I_nodeID].reward_sum / static_cast<double>(tree[I_nodeID].N);
      tree[J_nodeID].N += 1;
      tree[J_nodeID].reward_sum += reward_J;
      tree[J_nodeID].mu_hat = tree[J_nodeID].reward_sum / static_cast<double>(tree[J_nodeID].N);
      calc_LUCB();

      if(LCB[argmaxL] > UCB[argmaxU]){ // stop
        tree.calc_mu_hat();
        // std::vector<int> N(L_size);
        // for(int i=0; i<L_size; i++){
        //   N[i] = tree[tree.l_to_n[i]].N;
        // }
        // breakfunc();
        if(tree[argmaxL].mu_hat >= theta){
          return true;
        }
        else{
          return false;
        }
      }

      // if(t % 2000 == 0){
        
      //   std::vector<int> N(L_size);
      //   for(int i=0; i<L_size; i++){
      //     N[i] = tree[tree.l_to_n[i]].N;
      //   }
      //   std::cout << N << std::endl;
      //   std::cout << UCB << std::endl;
      //   std::cout << LCB << std::endl;
      //   // std::cout << "best: " << best << std::endl;
      //   // std::cout << "challenger: " << challenger << std::endl;

      //   breakfunc();

      // }


      // if(t > 10000) breakfunc();
      t++;
    }

    // breakfunc("error!");
    return false;
  }

  void print(){
    std::cout << "algorighm: P-track\n";
    std::cout << "N_size: " << this->N_size << "\n";
    std::cout << "L_size: " << this->N_size << "\n";
    std::cout << "theta: " << this->theta << "\n";
    std::cout << "delta: " << this->delta_coef << "e" << this->delta_exp << "\n";
    // std::cout << "nodes:\n"
    // for(int i=0; i<this->N_size; i++){
    //   this->tree[i].print();
    // }
    std::cout << std::endl;
  }
};


class Alg_RoundRobin : public Algorithm_base{
  public:
  int next_index;

  void set_param(const Problem &prob, const double delta_coef, const double delta_exp, const Experiment exp){
    this->N_size = prob.N_size;
    this->L_size = prob.L_size;
    this->tree.nodes = std::vector<Tree_Alg::Node>(N_size);
    this->tree.l_to_n = prob.tree.l_to_n;
    this->tree.n_to_l = prob.tree.n_to_l;


    for(int i=0; i<N_size; i++){
      this->tree[i].id = prob.tree[i].id;
      this->tree[i].depth = prob.tree[i].depth;
      this->tree[i].is_leaf = prob.tree[i].is_leaf;
      this->tree[i].type = prob.tree[i].type;
      this->tree[i].parent = prob.tree[i].parent;
      this->tree[i].child = prob.tree[i].child;
    }
    this->theta = prob.theta;
    this->delta_coef = delta_coef;
    this->delta_exp = delta_exp;

    for(int i=0; i<this->N_size; i++){
      this->tree[i].N = 0;
      this->tree[i].reward_sum = 0;
      this->tree[i].mu_hat = 0;
      this->tree[i].d_value = 0;
      // this->tree[i].sub_weight = std::vector<double>(this->L_size);
    }

    this->exp = exp;
  }

  void reset(){
    for(int i=0; i<this->N_size; i++){
      this->tree[i].N = 0;
      this->tree[i].reward_sum = 0;
      this->tree[i].mu_hat = 0;
      this->tree[i].d_value = 0;
      // this->tree[i].sub_weight = {};
    }
    this->next_index = 0;
  }

  double beta(int t){ 
    // return std::log(std::log(t+1) / this->delta);
    return std::log(std::log(t+1)) - std::log(this->delta_coef) - this->delta_exp * std::log(10);
  }

  // true: 停止, false: 継続
  bool is_stop(int t){
    if(this->exp == Experiment::SAMPLING_PROPORTION){
      if(t >= T_END) return true;
      else return false;
    }

    double b = beta(t);
    if(tree[0].Z >= b || -tree[0].Z >= b){
      return true;
    }
    else{
      return false;
    }
  }

  bool run(Oracle &oracle){
    // initialize
    this->reset();

    int t = 1;
    while(true){

      // draw leaf
      int reward = oracle.draw(this->next_index);
      Tree_Alg::Node &node = tree[tree.l_to_n[this->next_index]];
      node.N++;
      node.reward_sum += reward;
      node.mu_hat = node.reward_sum / static_cast<double>(node.N);
      tree.calc_mu_hat();
      tree.calc_Z(theta);

      this->next_index = (this->next_index+1) % this->L_size;
      
      // debug info
      std::vector<int> N(L_size);
      for(int i=0; i<L_size; i++){
        N[i] = tree[tree.l_to_n[i]].N;
      }
      
      if(is_stop(t)){
        // breakfunc();
        if(tree[0].mu_hat >= theta){
          return true;
        }
        else{
          return false;
        }
      }
      t++;
    }

    // breakfunc("error!");
    return false;
  }



  void print(){
    std::cout << "algorighm: P-track\n";
    std::cout << "N_size: " << this->N_size << "\n";
    std::cout << "L_size: " << this->N_size << "\n";
    std::cout << "theta: " << this->theta << "\n";
    std::cout << "delta: " << this->delta_coef << "e" << this->delta_exp << "\n";
    // std::cout << "nodes:\n"
    // for(int i=0; i<this->N_size; i++){
    //   this->tree[i].print();
    // }
    std::cout << std::endl;
  }

  void print_mu_hat(){
    // std::vector<double> mu_hat(L_size);
    // for(int i=0; i<L_size; i++){
    //   mu_hat[i] = tree[tree.l_to_n[i]].mu_hat;
    // }
    std::vector<double> mu_hat(N_size);
    for(int i=0; i<N_size; i++){
      mu_hat[i] = tree[i].mu_hat;
    }
    std::cout << "mu_hat: " << mu_hat << "\n";
    std::cout << std::endl;
  }

  void print_nodes(){
    for(int i=0; i<N_size; i++){
      tree[i].print();
    }
  }
};


void Oracle::record_diff(){
    // 選択比率の誤差
    double max_diff = 0;
    for(int i=0; i<L_size; i++){
      double diff = std::abs(optimal_weight[i] - static_cast<double>(N[i]) / static_cast<double>(N_total));
      max_diff = std::max(max_diff, diff);
    }
    this->sample_diff[(N_total / T_STEP) - 1] = max_diff;

    // weightの誤差
    max_diff = 0;
    for(int i=0; i<L_size; i++){
      double diff = std::abs(optimal_weight[i] - alg->tree[0].sub_weight[i]);
      max_diff = std::max(max_diff, diff);
    }
    this->weight_diff[(N_total / T_STEP) - 1] = max_diff;
  }


int main(int argc, char **argv) {
  Timer timer;
  std::string start_date = timer.now_str();

  Args args;
  args.init(argc, argv);
  // args.print();

  Problem prob;
  prob.read_probfile(args.probfile);
  // prob.print();
  // for(int i=0; i<prob.N_size; i++){
  //   prob.tree[i].print();
  // }

  if(args.theta > 0){
    prob.theta = args.theta;
  }


  Algorithm_base *alg;
  switch(args.algorithm){
    case D_TRACKING:
      alg = new Alg_Dtracking();
      break;
    case C_TRACKING:
      alg = new Alg_Ctracking();
      break;
    case P_TRACKING:
      alg = new Alg_Ptracking();
      break;
    case RANDOM_TRACKING:
      alg = new Alg_RandomTracking();
      break;
    case RATIO_D_TRACKING:
      alg = new Alg_WN_Dtracking();
      break;
    case UGAPE_MCTS:
      alg = new Alg_UGapE();
      break;
    case LUCB_MICRO:
      alg = new Alg_LUCBmicro();
      break;
    case ROUND_ROBIN:
      alg = new Alg_RoundRobin();
      break;
  }
  alg->set_param(prob, args.delta_coef, args.delta_exp, args.exp);
  // alg->print();
  // for(int i=0; i<alg->N_size; i++){
  //   alg->tree[i].print();
  // }

  Oracle oracle(prob, args.exp, alg);
  // oracle.print();

  std::ofstream ofs(args.outfile, std::ios::out);
  
  timer.start();
  // 実験1
  if(args.exp == Experiment::SAMPLING_PROPORTION){
    std::cout << "experiment 1: sampling proportion" << std::endl;

    // 比較用の真の最適重みを計算
    Tree_Alg tree;
    tree.load_prob(prob);
    tree.calc_mu_hat();
    tree.calc_d_value(prob.theta);
    calc_weight(tree, prob.theta);
    // for(int i=0; i<prob.N_size; i++){
    //   tree[i].print();
    // }
    // std::cout << tree[0].sub_weight << std::endl;
    oracle.set_optimal_weight(tree[0].sub_weight);


    Timer log_timer;
    int log_num = T_END / T_STEP;
    std::vector<double> sample_diff = std::vector<double>(log_num, 0);
    std::vector<double> sample_diff2 = std::vector<double>(log_num, 0);
    std::vector<double> weight_diff = std::vector<double>(log_num, 0);
    std::vector<double> weight_diff2 = std::vector<double>(log_num, 0);

    log_timer.start();
    for(int itr=1; itr<=args.itr_num; itr++){
      bool ans = alg->run(oracle);
      std::vector<double> &sdiff = oracle.get_sample_diff();
      std::vector<double> &wdiff = oracle.get_weight_diff();
      for(int i=0; i<log_num; i++){
        sample_diff[i] += sdiff[i];
        sample_diff2[i] += sdiff[i] * sdiff[i];
        weight_diff[i] += wdiff[i];
        weight_diff2[i] += wdiff[i] * wdiff[i];
      }

      oracle.reset_N();

      if(log_timer.elapsed_sec() > log_span_sec){
        std::cout << timer.now_str() << " " << itr << "/" << args.itr_num << std::endl;
        log_timer.start();
      }
    }
    std::vector<double> sdiff_avg(log_num);
    std::vector<double> sdiff_var(log_num);
    std::vector<double> sdiff_stddev(log_num);
    std::vector<double> wdiff_avg(log_num);
    std::vector<double> wdiff_var(log_num);
    std::vector<double> wdiff_stddev(log_num);
    for(int i=0; i<log_num; i++){
      sdiff_avg[i] = sample_diff[i] / args.itr_num;
      sdiff_var[i] = sample_diff2[i] / args.itr_num - sdiff_avg[i] * sdiff_avg[i];
      // 葉ノードが多いと、draw each leaf once中に出力ステップが来ることがある。
      // 選択順が固定なので分散は本来ちょうど0になるが、
      // このとき計算誤差で極小マイナス値になってしまうことがあるので0に逃がす。
      // 正の極小値にずれることもあるが、こちらは放置
      sdiff_var[i] = std::max(sdiff_var[i], 0.0); 
      sdiff_stddev[i] = std::sqrt(sdiff_var[i]);

      wdiff_avg[i] = weight_diff[i] / args.itr_num;
      wdiff_var[i] = weight_diff2[i] / args.itr_num - wdiff_avg[i] * wdiff_avg[i];
      // 葉ノードが多いと、draw each leaf once中に出力ステップが来ることがある。
      // 選択順が固定なので分散は本来ちょうど0になるが、
      // このとき計算誤差で極小マイナス値になってしまうことがあるので0に逃がす。
      // 正の極小値にずれることもあるが、こちらは放置
      wdiff_var[i] = std::max(wdiff_var[i], 0.0); 
      wdiff_stddev[i] = std::sqrt(wdiff_var[i]);
    }

    ofs << "start: " << start_date << "\n";
    ofs << "finish: " << timer.now_str() << " (" << timer.elapsed_str() << ")" << "\n";
    ofs << "probfile: " << args.probfile << "\n";
    ofs << "algorithm: " << args.algorithm << "\n";
    ofs << "itr: " << args.itr_num << "\n";
    ofs << "sample_proportion_diff_avg:\n";
    ofs << sdiff_avg << "\n";
    // ofs << "diff_var:\n";
    // ofs << sdiff_var << "\n";
    ofs << "sample_proportion_diff_stddev:\n";
    ofs << sdiff_stddev << "\n";
    ofs << "weight_diff_avg:\n";
    ofs << wdiff_avg << "\n";
    // ofs << "diff_var:\n";
    // ofs << sdiff_var << "\n";
    ofs << "weight_diff_stddev:\n";
    ofs << wdiff_stddev << "\n";
    ofs << std::flush;
  }
  // 実験2
  else if(args.exp == Experiment::STOPPING_TIME){
    std::cout << "experiment 2: stopping time" << std::endl;
    std::ofstream out_each_itr;
    
    if(record_each_iteration && args.enable_second_output){
      out_each_itr.open(args.outfile2, std::ios::out);
    }

    Timer log_timer;
    double total_time = 0; // sum of the stopping times
    double total_time2 = 0; // sum of the squared stopping times
    int correct_count = 0; // 正解数

    log_timer.start();
    for(int itr=1; itr<=args.itr_num; itr++){
      bool ans = alg->run(oracle);
      if(ans == prob.is_win) correct_count++;

      double tau = oracle.get_N_total();
      total_time += tau;
      total_time2 += tau * tau;
      oracle.reset_N();

      if(record_each_iteration){
        out_each_itr << tau << "\n";
      }

      if(log_timer.elapsed_sec() > log_span_sec){
        std::cout << timer.now_str() << " " << itr << "/" << args.itr_num << std::endl;
        log_timer.start();
      }
    }
    if(record_each_iteration){
      out_each_itr.close();
    }

    double tau_avg = total_time / args.itr_num;
    double tau_var = total_time2 / args.itr_num - tau_avg * tau_avg;

    ofs << "start: " << start_date << "\n";
    ofs << "finish: " << timer.now_str() << " (" << timer.elapsed_str() << ")" << "\n";
    ofs << "probfile: " << args.probfile << "\n";
    ofs << "theta: " << prob.theta << "\n";
    ofs << "delta: " << args.delta_coef << "e" << args.delta_exp << "\n";
    ofs << "algorithm: " << args.algorithm << "\n";
    ofs << "itr: " << args.itr_num << "\n";
    ofs << "correct_count: " << correct_count << "\n";
    ofs << "tau_avg: " << tau_avg << "\n";
    ofs << "tau_var: " << tau_var << "\n";
    ofs << "tau_stddev: " << std::sqrt(tau_var) << "\n";
    ofs << std::flush;
  }

  ofs.close();

  std::cout << "finish\n";
  std::cout << "-> " << args.outfile << std::endl;

  return 0;
}



