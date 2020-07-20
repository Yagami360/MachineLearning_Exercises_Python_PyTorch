# GCN_sample_classication
cora dataset を使用して、論文中の頻出単語情報からその論文の論文カテゴリを予想する Graph Convolutional Networks (GCN) ベースの分類器

- 参考コード
    - [GitHub/tkipf/pygcn](https://github.com/tkipf/pygcn)

## ■ 使用法

- 学習処理
  ```sh
  # （例）
  $ python train.py \
    --exper_name graph_convolutional_networks \
    --dataset_dir ${CORA_DATASET_DIR} \
    --n_epoches 200
  ```

- TensorBoard
  ```sh
  $ tensorboard --logdir ${TENSOR_BOARD_DIR} --port ${AVAILABLE_POOT}
  ```

  ```sh
  #（例）
  $ tensorboard --logdir tensorboard --port 6006
  ```

## ■ データセット
cora dataset を使用してグラフ畳み込みネットワークを学習。
このモデルでは、論文中の頻出単語（特徴量）から論文カテゴリを予想するのが目的

- Download 先 : www.research.whizbang.com/data

- cora.cites : 2708 個の論文の引用被引用関係の辺情報（＝隣接行列）を格納
  - 隣接行列 : 2708 x 2708
  ```python
  # target : 被引用論文 / データセット中には含まれない列名 
  # source : 引用論文 / データセット中には含まれない列名
    target   source
  0      35     1033
  1      35   103482
  2      35   103515
  3      35  1050679
  4      35  1103960
  ```

- cora.content : 論文中の頻出単語情報と論文カテゴリを格納
  ![image](https://user-images.githubusercontent.com/25688193/87515953-9804fd00-c6b7-11ea-9be1-25745f4b67ad.png)
  - `w_0` ~ `w_1432` : その単語が頻出しているかの値（0:頻出でない、1:頻出）（1433 個）/ データセット中には含まれない列名
  - `subject` : その論文のカテゴリ（７個）/ データセット中には含まれない列名
    ```python
    'Case_Based',
    'Genetic_Algorithms',
    'Neural_Networks',
    'Probabilistic_Methods',
    'Reinforcement_Learning',
    'Rule_Learning',
    'Theory'
    ```

## ■ コードの実行結果

### ◎ 正解率のグラフ


### ◎ 損失関数のグラフ


