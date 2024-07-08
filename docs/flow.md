
### アルゴリズムのフローとMCTSrの各メソッドの対応
| Mermaid工程 | 関連するメソッド | 目的 | 詳細 |
|-------------|------------------|------|------|
| Initialize MCTSr | `__init__` | アルゴリズムの実行に必要な初期設定を行う | - 問題、正解、最大反復回数などの基本パラメータを設定<br>- 探索ノード、報酬、ヒント、UCB値などのデータ構造を初期化<br>- アルゴリズムの動作を制御する閾値や制限を設定 |
| Generate initial weak answer | `get_weak_answer` | 問題に対する初期の回答を生成する | - 言語モデルを使用して、与えられた問題に対する最初の回答を作成<br>- この回答は「弱い」または「改善の余地がある」と想定される<br>- 探索の出発点となる基準を設定 |
| Calculate initial reward | `sampling_reward`, `cal_reward` | 初期回答の品質を評価し、数値化する | - 生成された初期回答の正確さ、完全性、適切性を分析<br>- -100から100の範囲でスコアを割り当て<br>- 初期状態の評価基準を確立し、今後の改善の基準点を設定 |
| Update UCB values | `update_ucb`, `compute_ucb` | 各ノードの上限信頼度（UCB）値を更新し、探索と活用のバランスを取る | - 各ノードの報酬と訪問回数に基づいてUCB値を計算<br>- 木構造全体にわたってUCB値を更新（バックプロパゲーション）<br>- 次の探索で最も有望なノードを特定するための基準を提供 |
| Filter mature nodes | `filter_mature_node` | さらなる探索が有益な可能性のあるノードを識別する | - 十分に展開されていないノードや、さらなる改善の可能性があるノードを選別<br>- 探索の効率を高め、計算リソースを効果的に利用 |
| Select best node from UCB | `get_best_explore_from_ucb` | UCB値に基づいて、次に探索するべき最も有望なノードを選択する | - フィルタリングされたノードの中から最高のUCB値を持つノードを特定<br>- 探索と活用のバランスを取りながら、最も改善の可能性が高いノードを選択 |
| Sample reward for selected node | `sampling_reward`, `cal_reward` | 選択されたノードの現在の品質を評価する | - 選択されたノードの回答に対して新たに報酬を計算<br>- ノードの現在の状態を評価し、以前の評価との比較のための基準を提供 |
| Generate weak hints | `get_weak_hints` | 選択されたノードの回答を改善するためのヒントを生成する | - 回答の弱点や改善が必要な点を特定<br>- 具体的で建設的なフィードバックを提供<br>- 次のステップでの回答改善のための指針を提供 |
| Generate better answer | `get_better_answer` | 生成されたヒントに基づいて、改善された回答を作成する | - 前のステップで生成されたヒントを考慮に入れる<br>- 言語モデルを使用して、より正確で完全な回答を生成<br>- 回答の質を段階的に向上させる |
| Generate structured feedback | `generate_structured_feedback` | 改善された回答に対してさらに詳細で構造化されたフィードバックを提供する | - 回答の様々な側面（推論過程、事実の正確さ、明確さなど）を評価<br>- 具体的で行動可能なフィードバックを構造化された形式で提供<br>- 次のステップでのさらなる改善のための詳細な指針を提供 |
| Refine answer with feedback | `refine_answer_with_feedback` | 構造化されたフィードバックを基に回答をさらに洗練させる | - 提供されたフィードバックの各ポイントに対処<br>- 回答の質をさらに向上させ、より完璧な解答に近づける<br>- 改善プロセスの反復を通じて、回答の継続的な向上を図る |
| Update data structures | `add_to_hints_bank`, `add_to_childs`, `add_to_hints_reward_imp_bank` | アルゴリズムの内部状態を更新し、新しい情報を記録する | - 新しく生成されたヒント、改善された回答、報酬などを適切なデータ構造に追加<br>- 探索木の構造を更新（新しい子ノードの追加など）<br>- 将来の決定と評価のための履歴情報を維持 |
| Check termination condition | `check_termination` | アルゴリズムを終了するべきかどうかを判断する | - 正解に到達したかどうかを確認<br>- 最大反復回数に達したかを確認<br>- 改善率が閾値を下回ったかどうかを評価<br>- 探索の深さが制限を超えていないかを確認 |
| Select best final answer | (メインループ内で実装) | 探索プロセス全体を通じて得られた最良の回答を選択する | - すべての生成された回答の中から最高の報酬を持つものを特定<br>- 必要に応じて、複数の高品質な回答の中から最適なものを選択<br>- アルゴリズムの最終的な出力となる回答を決定 |
| Process and save results | `process_example`, `main` | アルゴリズムの実行結果を処理し、保存する | - 最終的な回答、探索の履歴、各ステップでの中間結果などを構造化<br>- 結果を分析可能な形式（例：JSON）で保存<br>- 将来の分析や改善のために、プロセス全体の詳細な記録を維持 |****

### アルゴリズムのフローを表す mermaid グラフ
```mermaid
graph TD
    A[Start] --> B[Initialize MCTSr]
    B --> C[Generate initial weak answer]
    C --> D[Calculate initial reward]
    D --> E[Update UCB values]
    E --> F{Max iterations reached?}
    F -->|No| G[Filter mature nodes]
    G --> H[Select best node from UCB]
    H --> I[Sample reward for selected node]
    I --> J[Generate weak hints]
    J --> K[Generate better answer]
    K --> L[Generate structured feedback]
    L --> M[Refine answer with feedback]
    M --> N[Update data structures]
    N --> O[Check termination condition]
    O -->|Not met| E
    O -->|Met| P[Select best final answer]
    F -->|Yes| P
    P --> Q[Process and save results]
    Q --> R[End]
```

### 報酬サンプリングプロセスの図解
```mermaid
graph TD
    A["回答 a"] --> B{"サンプリング"}
    B -->|"1回目"| C1["報酬 R1"]
    B -->|"2回目"| C2["報酬 R2"]
    B -->|"3回目"| C3["報酬 R3"]
    B -->|"..."| C4["..."]
    B -->|"N回目"| C5["報酬 RN"]
    C1 --> D["報酬集合 R_a"]
    C2 --> D
    C3 --> D
    C4 --> D
    C5 --> D
    D --> E{"Q値計算"}
    E --> F["Q(a) = f(R_a)"]
    F --> G["探索と改善"]
```
#### 詳細
![sampling](./img/sampling.png)

### 報酬Rの決まり方
```mermaid
graph TD
    A["質問と回答"] --> B{"cal_reward メソッド"}
    B --> C["DetailedCritique モデル"]
    C --> D["スコア生成 (-100 ~ +100)"]
    D --> E{"スコア > 95?"}
    E -->|"Yes"| F["スコア調整: 95 + (スコア - 95) * 0.1"]
    E -->|"No"| G["スコアをそのまま使用"]
    F --> H["最終的な報酬 R"]
    G --> H
    H --> I["報酬サンプルの集合に追加"]
    I --> J{"Q値計算"}
    J --> K["Q(a) = (min(R) + avg(R)) / 2"]
    
    style A fill:#f9f,stroke:#333,stroke-width:2px
    style B fill:#bbf,stroke:#333,stroke-width:2px
    style C fill:#bfb,stroke:#333,stroke-width:2px
    style D fill:#fbb,stroke:#333,stroke-width:2px
    style E fill:#ff9,stroke:#333,stroke-width:2px
    style H fill:#f96,stroke:#333,stroke-width:2px
    style K fill:#9ff,stroke:#333,stroke-width:2px
```

###  バックプロパゲーションの概念図
![Backpropagation図](./img/backpropagation.png)


