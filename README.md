- モデルの目標を明確に説明（なぜこれを選んだのか）
- 何が難しくどのように改善したか
- NotebookLMを使用する
- コードの改善にはGoogle Colabを使用する

論文1「[Revisiting Deep Learning Models for Tabular Data](http://ar5iv.labs.arxiv.org/html/2106.11959)」
Transformerの原理を表形式データに応用する基礎理論。

論文2「[Focal Loss for Dense Object Detection」 (Lin et al., 2017)](http://arxiv.org/pdf/1708.02002)」
標準的な損失関数（Cross Entropy）を使用すると、モデルは「とりあえず全部『多数派』と答えておけば正解率が高くなる」という安易な学習に陥りやすくなります。「背景（多数）」と「物体（少数）」の不均衡を解決する画像処理の理論を、信用リスクの「返済（多数）」と「デフォルト（少数）」に応用します。



**ポートフォリオテーマ**
「**FinTechにおける戦略的AI：FT-Transformerと期待利益最大化による信用リスクスコアリング**」 (Strategic AI in FinTech: Profit-Driven Credit Scoring with FT-Transformer)

コンセプト: 単に「予測精度（AUC）」を競うのではなく、金融ビジネスの本質である**「期待利益（Expected Profit）」の最大化**を目的関数に据えます。技術的には、GBDT（決定木）が支配的なこの領域で、最新の深層学習（FT-Transformer）と不均衡データ対策（Focal Loss）を適用し、あなたの「エンジニアリング能力」と「ビジネス視点」の両方をアピールします。




Week 1: リレーショナル特徴量エンジニアリング (Data Engineering)

* 課題: Home Creditデータは7つのテーブル（1対多の関係）で構成されているが、Transformerは単一の入力（フラットなテーブル）を必要とする。
* タスク:bureau (他社借入) や installments_payments (返済履歴) などの子テーブルを、親ID (SK_ID_CURR) ごとに集約する。集約関数：平均 (Mean)、最大 (Max)、合計 (Sum)、分散 (Var) を使用し、顧客1人につき1行のデータを作成する。カテゴリ変数はLabel Encodingを行い、数値変数は正規化（Standardization）する。

Week 2: FT-Transformerの実装 (Model Implementation)
* 課題: 数値データをTransformerが扱える形式（トークン）に変換する必要がある。
* タスク:PyTorchを用いて Feature Tokenizer を実装（各数値特徴量を線形層で埋め込みベクトル化）。Transformer Encoder層（Multi-Head Attention + Feed Forward）を構築。ベースラインとして標準的なBCE Lossで学習させ、XGBoostと比較できる程度のAUCが出ることを確認する。

Week 3: 利益最大化損失関数の実装 (Profit Maximization)
* 課題: デフォルトを見逃すコスト（元本毀損）は、誤って貸さないコスト（金利機会損失）よりも遥かに大きい。
* タスク:Focal Loss を損失関数として実装する。パラメータ $\alpha$（不均衡調整）と $\gamma$（難易度フォーカス）を調整し、デフォルト予備軍（Hard Samples）の検出力を強化する。検証用データで、AUCだけでなくRecall（再現率）の変化をモニタリングする。

Week 4: 実験とビジネス評価 (Evaluation)
* 課題: 技術的な指標（AUC）をビジネス価値（金額）に翻訳する。
* タスク:利益スコア (Profit Score) の定義と計算：$$\text{Profit} = (TP \times 0) - (FN \times \text{Principal}) + (TN \times \text{Interest}) - (FP \times \text{Interest})$$Focal Lossを用いたモデルが、通常のモデルと比較してどれだけ「損失回避（Profit）」に貢献したかを可視化する。


Week 5: レポート執筆と倫理的考察 (Reporting)
* 課題: 技術的成果をアカデミックかつ実務的な文脈でまとめる。
* タスク:「なぜ画像処理の損失関数（Focal Loss）を金融に応用したのか」という技術的ナラティブを記述。EU AI Act（欧州AI法） への言及：信用スコアはハイリスクAIに分類されるため、モデルの透明性（Attention Mapによる可視化）や公平性への配慮について考察を加える。GithubでCI/CDパイプラインを構築し、ユニットテストを実行できるようにする
