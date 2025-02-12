# how to coding 
このセクションでは、まず図1に示すMCTSrの主な構造を説明します。次に、各コンポーネントの詳細を説明します。MCTSrの主なワークフローは以下のように構成されています:

- 初期化: モデルの過適合傾向を最小限に抑えるために、ナイーブなモデル生成回答またはダミー応答(例: 'わかりません')を使用してルートノードが確立されます。
- 選択: アルゴリズムは価値関数 Q を使用して完全に展開されていないすべての回答をランク付けし、貪欲戦略を用いてさらなる探索と改善のために最高値のノードを選択します。
- 自己改善: 選択された回答 a は自己改善フレームワーク(Madaan et al., 2023)を使用して最適化されます。最初に、モデルはフィードバック m を生成し、改善プロセスを導いて強化された回答 a' を生成します。
- 自己評価: 改善された回答は、報酬値をサンプリングし Q 値を計算するためにスコア付けされます。これには、モデルの自己報酬フィードバックと、スコアリングの信頼性と公平性を確保するための厳格なスコアリング基準や満点の抑制などの制約が含まれます。
- バックプロパゲーション: 改善された回答の値は、ツリーの価値情報を更新するために親ノードと他の関連ノードに後方伝播されます。子ノードの Q 値が変更された場合、親ノードの Q が更新されます。
- UCT更新: すべてのノードの Q 値が更新された後、さらなる拡張または選択のための候補ノードのコレクション C を特定し、次の選択段階のためにすべてのノードの UCT 値を更新するために UCT 更新式を使用します。

アルゴリズムは、ロールアウト制約や最大探索深度を含む終了条件 T が満たされるまで、これらの段階を繰り返します。継続的に回答の質を改善し、新しい可能性を探索します。

## 3.1 自己改善

自己改善プロセスでは、モデルは問題 P に対する回答 a を最適化するために、マルチターンの対話改善プロンプトによって誘導されます。最初に、モデルは a に関する反省的または批判的なコメント m を生成します。その後、m に導かれて、モデルは a を修正して改善されたバージョン a' を生成します。この反復的な改善は、構造化されたフィードバックを活用して回答の進化を促進し、回答の質を向上させます。
    
## 3.2 自己評価
    
問題 P の改善プロセスにおいて、回答 a の Q 値は、a からその書き直し形式への遷移のマルコフ性により、a をさらに改善してより優れた回答にする期待品質として定義されます。従来のMCTSで Q(s, a) が状態 s での行動 a の価値を推定するのとは異なり、ここでの Q(a) は a に帰属する報酬関数値の複数のサンプリングから導出されます。

モデルは、a の報酬を推定するために自己報酬法を使用し、-100から100の範囲で報酬スコアを提供することが要求されます。制約がない場合、モデルの報酬傾向が過度に滑らかになり、実践では回答間の比較的な区別が欠如することがわかりました。これに対処するために、3つの制約が設計されました:

- プロンプト制約: モデルは報酬スコアリング中に最も厳格な基準を遵守しなければなりません。
- 満点抑制: モデルは完全なフィードバックスコアを提供しないよう指示されます。95を超える報酬は過剰なスコアを抑制するために一定量減少されます。
- 繰り返しサンプリング: 探索木のノードへの各訪問には、自己評価の信頼性を高めるためにノードの報酬の繰り返しサンプリングが含まれます。ノードの子ノードに対して報酬サンプリングが実行される際、報酬サンプリングのサンプルサイズを増やすために親ノードに対しても報酬サンプリングが実行されることに注意してください。

サンプリング後、a の Q 値が計算されます。自己報酬関数の平滑化傾向に対抗するために、期待報酬に最小値制約が追加され、回答品質の推定がさらに改善されます。

Q(a) = 1/2 * (min R_a + 1/|R_a| * Σ(i=1 to |R_a|) R_a^i)

ここで、Q(a) は回答 a の品質値、R_a は a の報酬サンプルのセット、min R_a は R_a の最小報酬、|R_a| はサンプル数、Σ(i=1 to |R_a|) R_a^i は R_a のすべての報酬の合計です。この式は、報酬の最小値と平均値を平均することで Q(a) を計算し、最悪のケースと平均的な結果のバランスを取ります。
    
## 3.3 バックプロパゲーション
    
すべての葉ノードの報酬値のサンプリングと Q 値の更新が完了した後、この変更を親ノードと祖先ノードに伝播します。この更新プロセス中に、ノード a の子ノードセット Children(a) の任意の要素の Q 関数値が変更された場合、ノードの Q 関数値は以下のように更新されます:

Q'(a) = 1/2 * (Q(a) + max(i∈Children(a)) Q(i))

ここで、Q'(a) は子ノードからの影響を考慮した回答 a の更新された品質値、Q(a) は報酬サンプリングのみを考慮したナイーブな品質値、max(i∈Children(a)) Q(i) は a の子ノードの中で最高の品質値を表します。この式は、現在の値と後続の子ノードからの最良の可能な結果を平均することで Q(a) を改善します。
    
## 3.4 UCTの更新と選択
    
ツリー内のすべてのノードの Q 値を更新した後、次のラウンドの選択のための選択フェーズに進みます。このプロセスには以下のステップが含まれます:

候補ノードの選択: 数学問題の改善プロセスのマルコフ性を活用し、すべての葉ノードと完全に展開されていないノードの選択に焦点を当て、改善パスの履歴を無視することが可能です。このパス独立性により、問題が簡素化されます。ノードを選択する際にルートノードから開始する必要はなく、ツリー内のノードを階層順に走査します。

しかし、このタスクでポリシーとして機能する大規模言語モデル(LLM)は、任意の回答状態 a に対して無限数の改善アクション m を生成できるため、各ノードは潜在的に無限のアクションセットに直面します。そこで、ベイズ最適化の期待改善の概念を参考に、「完全展開」を決定するための2つの基準を提案します:

- ノードの子の数が事前定義された制限に達している。そして、
- 少なくとも1つの子ノードの Q 値がノードの値を超えている。

これらの基準に基づいて、さらなる拡張または選択のための候補ノードのコレクション C を特定します。この戦略は、後続の探索でより高い価値の回答をもたらす可能性のあるノードを正確に定義するのに役立ち、全体的な探索効率と結果の品質を向上させます。

UCT更新: AlphaGoから着想を得て、ノードの探索と活用のバランスを取るためにUCB-1法を用いたUCTを使用します。候補セット C のノード a に対して、その UCT_a 値は以下のようになります:

UCT_a = Q(a) + c * sqrt(ln(N(Father(a)) + 1) / (N(a) + ε))

ここで、Q(a) は回答 a の Q 値、N(·) は与えられたノードの総訪問回数、c は活用と探索のバランスを取る定数、ε はゼロ除算を避けるための小さな定数です。

ソートと選択: 候補セット C の UCT 値に従って、貪欲サンプリングまたは重要度サンプリングを通じて改善プロセスを探索するための最適なノードを選択できます。
    
## 3.5 終了関数
    
MCTSrアルゴリズムでは、探索終了関数基準 T はいくつかの条件から導出できます:

早期停止: 探索結果の改善が減少した場合、または連続した探索が繰り返しの結果をもたらす場合に終了します。

探索制約: ロールアウト数が事前定義された制限に達した場合、またはツリー内の1つ以上のノードが最大深度制約を満たした場合に探索が終了します。

言語モデルのロジットに基づく高度な基準: 言語モデルのロジットから導出された事前定義された指標に基づいて探索が終了します。

終了関数条件 T が満たされたら、Q 値や他の条件に基づいてツリーノードから最良の回答を収集できます。