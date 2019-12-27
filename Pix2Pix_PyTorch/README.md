# Pix2Pix_PyTorch
pix2pix の PyTorch での実装。<br>
pix2pix によるセマンティックセグメンテーションを利用して、衛星画像から地図を生成する。<br>

- 参考コード
    - [PyTorch-GAN/implementations/pix2pix/](https://github.com/eriklindernoren/PyTorch-GAN/tree/master/implementations/pix2pix)

## ■ 項目 [Contents]
1. [動作環境](#動作環境)
1. [使用法](#使用法)
1. [コードの実行結果](#コードの実行結果)
1. [背景理論](https://github.com/Yagami360/My_NoteBook/blob/master/%E6%83%85%E5%A0%B1%E5%B7%A5%E5%AD%A6/%E6%83%85%E5%A0%B1%E5%B7%A5%E5%AD%A6_%E6%A9%9F%E6%A2%B0%E5%AD%A6%E7%BF%92_%E7%94%9F%E6%88%90%E3%83%A2%E3%83%87%E3%83%AB.md#pix2pix)


## ■ 動作環境

- Mac OS / ubuntu server
- Python : 3.6
- Anaconda : 5.0.1
- PyTorch : 1.1.0

## ■ 使用法

- 使用データ（航空写真と地図画像）<br>
  https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/maps.tar.gz<br>
  よりダウンロード後、解凍。

- 学習処理
  ```sh
  # （例２） train Pix2Pix for air map datset using GPU0
  # when save datset dataset/maps dir from https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/maps.tar.gz
  $ python train.py \
    --exper_name Pix2Pix_train \
    --dataset_dir dataset/maps \
    --image_size 256 \
    --unetG_dropout 0.5 --networkD_type PatchGAN
  ```

- 推論処理（実装中...）
  ```sh
  $ python test.py
  ```

- TensorBoard
  ```sh
  $ tensorboard --logdir ${TENSOR_BOARD_DIR} --port ${AVAILABLE_POOT}
  ```

  ```sh
  #（例）
  $ tensorboard --logdir tensorboard --port 6006
  ```

<a id="コードの実行結果"></a>

## ■ コードの実行結果

### ◎ 損失関数のグラフ

- 識別器側<br>
<br>
- 生成器側<br>
<br>
  - 学習用データセット（緑）
  - テスト用データセット（灰色）

### ◎ 生成器からの生成画像


## ■ デバッグ情報
