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
- tensorboard : 1.13.1
- tensorboardx : 1.9
- tqdm

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

### ◎ 生成器からの生成画像

- Epoches : 10<br>
  ![fake_image_epoches10_batchAll](https://user-images.githubusercontent.com/25688193/71606211-51840100-2bb2-11ea-8feb-6b3603abce70.png)<br>
- Epoches : 50<br>
  ![fake_image_epoches50_batchAll](https://user-images.githubusercontent.com/25688193/71606207-51840100-2bb2-11ea-82c6-19a80d99d48d.png)<br>
- Epoches : 100<br>
  ![fake_image_epoches99_batchAll](https://user-images.githubusercontent.com/25688193/71606210-51840100-2bb2-11ea-8543-593afadbcc1e.png)<br>
- Epoches : 1 ~ 100<br>
  ![fake_image_epoches99](https://user-images.githubusercontent.com/25688193/71606208-51840100-2bb2-11ea-9ab8-3796264c022e.gif)<br>

→ UNet による生成画像は全体的にぼやけた画像になっていたが、生成器の構造が同じ UNet 構造となっている Pix2Pix では全体的にくっきりとした画像が生成出来ていることに注目。

### ◎ 損失関数のグラフ

- 識別器側 : Epoches 1 ~ 100<br>
  ![image](https://user-images.githubusercontent.com/25688193/71606246-90b25200-2bb2-11ea-9792-e5e2d3e6dfc2.png)<br>
- 生成器側 : Epoches 1 ~ 100<br>
  ![image](https://user-images.githubusercontent.com/25688193/71606259-a758a900-2bb2-11ea-8721-6b2643d59321.png)<br>
  - 青色：学習用データセット（ミニバッチ単位）
  - 茶色：テスト用データセット（テスト用データ全体）


## ■ デバッグ情報
