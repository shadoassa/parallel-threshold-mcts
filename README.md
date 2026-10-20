# 閾値MCTSのMPI並列化とその理論的解析

先行研究 "An Efficient Algorithm for Thresholding Monte Carlo Tree Search" (NeurIPS 2026 投稿版) の実装をベースに, RD-Tracking-MCTSをMPIで並列化し, 理論的な性能改善を示すことを目的とするプロジェクト.

## プロジェクト構成

```
parallel-threshold-mcts/
├── src/
│   ├── origin/          # 先行研究の実装 (逐次版)
│   └── test/            # MPI動作確認用デモ
└── docs/
    └── literature/      # 参照論文
```

## 先行研究実装 (`src/origin/`) の実行フロー

### 全体パイプライン

```mermaid
flowchart TD
    A[make_cmd_randomtree.py\n問題生成コマンドを生成] -->|sh tmp.sh| B[gen_prob_review.py\nランダム木を生成]
    B -->|prob_randomtree/prob_i.txt| C[(問題ファイル群\n500本)]

    C --> D[make_cmd_randomtree2.py\n実験コマンドを生成]
    D -->|tmp.sh| E[delay_exec.py\n最大100並列でコマンドを実行]
    E -->|./experiment ...| F[experiment\nC++実装 各アルゴリズムを実行]
    F -->|result_*.txt| G[(実験結果ファイル群)]

    G --> H[calc_average_randomtree.py\n停止時刻の平均・下界比を集計]
    H -->|数値をコピー| I[graph_randomtree.py\nグラフを描画]
    I --> J[fig_randomtree.png]
```

### `experiment.cpp` 内部構造

```mermaid
classDiagram
    class Algorithm_base {
        <<abstract>>
        +int N_size
        +int L_size
        +Tree_Alg tree
        +double theta
        +set_param(prob, delta_coef, delta_exp, exp)
        +run(oracle) bool
    }

    class Alg_Dtracking {
        +run(oracle) bool
    }
    class Alg_Ctracking {
        +vector weight_sum
        +run(oracle) bool
    }
    class Alg_WN_Dtracking {
        +run(oracle) bool
    }
    class Alg_RandomTracking {
        +run(oracle) bool
    }
    class Alg_Ptracking {
        +vector beta_a
        +vector beta_b
        +Tree_Alg po_tree
        +run(oracle) bool
    }
    class Alg_UGapE {
        +vector UCB
        +vector LCB
        +run(oracle) bool
    }
    class Alg_LUCBmicro {
        +vector UCB
        +vector LCB
        +run(oracle) bool
    }
    class Alg_RoundRobin {
        +int next_index
        +run(oracle) bool
    }

    Algorithm_base <|-- Alg_Dtracking : D-Tracking
    Algorithm_base <|-- Alg_Ctracking : C-Tracking
    Algorithm_base <|-- Alg_WN_Dtracking : Ratio-based D-Tracking
    Algorithm_base <|-- Alg_RandomTracking : Random-Tracking
    Algorithm_base <|-- Alg_Ptracking : P-Tracking
    Algorithm_base <|-- Alg_UGapE : UGapE-MCTS
    Algorithm_base <|-- Alg_LUCBmicro : LUCB-micro
    Algorithm_base <|-- Alg_RoundRobin : Round Robin

    class Tree_Alg {
        +vector nodes
        +vector l_to_n
        +vector n_to_l
        +calc_mu_hat()
        +calc_d_value(theta)
        +calc_Z(theta)
        +update_mu_hat(reward, leaf_id)
    }

    class Oracle {
        +int L_size
        +vector true_mu
        +draw(leaf_id) int
        +reset_N()
    }

    class Problem {
        +int N_size
        +int L_size
        +Tree_Prob tree
        +double theta
        +bool is_win
        +read_probfile(filename)
    }

    Algorithm_base *-- Tree_Alg
    Algorithm_base ..> Oracle : uses
    Algorithm_base ..> Problem : set_param
```

### 1回の試行における処理の流れ

```mermaid
sequenceDiagram
    participant main
    participant Alg as Algorithm
    participant Oracle

    main->>Alg: set_param(prob, delta, exp)
    loop itr_num 回
        main->>Alg: run(oracle)
        Alg->>Oracle: draw(leaf_id) × L_size 回 (初期化)
        loop 停止条件を満たすまで
            Alg->>Alg: calc_d_value(theta)
            Alg->>Alg: calc_weight(tree, theta)
            Alg->>Oracle: draw(next_leaf_id)
            Oracle-->>Alg: reward (0 or 1)
            Alg->>Alg: update_mu_hat, calc_Z
            Alg->>Alg: is_stop(t)?
        end
        Alg-->>main: win/lose
        main->>Oracle: reset_N()
    end
    main->>ofs: 結果ファイルに書き出し
```

## 実装済みアルゴリズム一覧 (先行研究)

| アルゴリズム | クラス名 | 概要 |
|---|---|---|
| D-Tracking | `Alg_Dtracking` | 最適重みとの差 $t \cdot w_\ell^* - N_\ell(t)$ が最大の腕を選択 |
| C-Tracking | `Alg_Ctracking` | 重みの累積和を追跡し, 最適配分に近づける |
| Ratio-based D-Tracking | `Alg_WN_Dtracking` | $w_\ell / N_\ell$ が最大の腕を選択 |
| Random-Tracking | `Alg_RandomTracking` | 最適重みに比例した確率でランダムに腕を選択 |
| P-Tracking | `Alg_Ptracking` | Thompson Sampling (Beta分布) で重みを推定 |
| UGapE-MCTS | `Alg_UGapE` | UCB/LCBを再帰的に伝播させ Best/Challenger を比較 |
| LUCB-micro | `Alg_LUCBmicro` | 単調性制約付きUCB/LCBを使用するLUCBの木版 |
| Round Robin | `Alg_RoundRobin` | 全葉ノードを順番に選択するベースライン |

## 停止条件

$$Z_{s_0}(t) \geq \beta(t), \quad \beta(t) = \log\log(t+1) + \log(1/\delta)$$

$Z_{s_0}(t)$ は根ノードの証拠統計量 (KL情報量の木上での集約値).

## 開発目標

- [ ] RD-Tracking-MCTSのMPI並列化実装
- [ ] Virtual Loss によるプロセス分散
- [ ] 並列版と逐次版の停止時刻比較実験
- [ ] 並列化による理論的サンプル複雑性の改善解析
