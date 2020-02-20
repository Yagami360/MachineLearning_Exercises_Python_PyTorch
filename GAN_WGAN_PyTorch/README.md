# GAN_WGAN_PyTorch
WGAN の PyTorch での実装。

- 参考コード
  - [martinarjovsky/WassersteinGAN](https://github.com/martinarjovsky/WassersteinGAN)

## ■ 項目 [Contents]
1. [動作環境](#動作環境)
1. [使用法](#使用法)
1. [コードの実行結果](#コードの実行結果)
1. [背景理論](https://github.com/Yagami360/My_NoteBook/blob/master/%E6%83%85%E5%A0%B1%E5%B7%A5%E5%AD%A6/%E6%83%85%E5%A0%B1%E5%B7%A5%E5%AD%A6_%E6%A9%9F%E6%A2%B0%E5%AD%A6%E7%BF%92_%E7%94%9F%E6%88%90%E3%83%A2%E3%83%87%E3%83%AB.md#WGAN)

## ■ 動作環境

- Ubuntu : 16.04
- Python : 3.6
- Anaconda : 5.0.1
- PyTorch : 1.x 系
- tensorboardx :
- tqdm :
- imageio :

## ■ 使用法

- 学習処理
  ```sh
  # （例１） train WGAN for MNIST datset using GPU0
  $ python train.py \
    --exper_name WGAN_train \
    --dataset mnist --image_size 64
  ```

- 推論処理（学習済みモデルから画像生成）
  ```sh
  $ python test.py --load_checkpoints_dir ${LOAD_CHAECKPOINTS_DIR}
  ```

- 推論処理（学習済みモデルからモーフィング動画の作成）
  ```sh
  $ python test_morphing.py --load_checkpoints_dir ${LOAD_CHAECKPOINTS_DIR}
  ```
  ```sh
  # （例）
  $ python test_morphing.py \
      --exper_name WGAN_test_morphing \
      --load_checkpoints_dir checkpoints/WGAN_train_D_NonBatchNorm_Opt_RMSprop_Epoch50_191228_1 \
      --fps 30 --codec gif
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

#### ☆ 実行条件１（識別器の BatchNorm あり）

- Epoches : 10
  ![fake_image_epoches10_batchAll](https://user-images.githubusercontent.com/25688193/71560371-67a79980-2aac-11ea-828d-1308c8ec25fa.png)<br>

- Epoches : 50
  ![fake_image_epoches49_batchAll](https://user-images.githubusercontent.com/25688193/71566416-c5fb6900-2afa-11ea-8da3-65d55773d072.png)<br>

- Epoches : 1 ~ 50<br>
  ![fake_image_epoches49](https://user-images.githubusercontent.com/25688193/71566415-c5fb6900-2afa-11ea-88e9-225a60578a69.gif)<br>

  → 学習は安定化するものの、生成画像の品質はそれほど高くない印象<br>
  → 又、学習時間が長いという問題もあった。

- 学習済みモデルからのモーフィング動画<br>
  ![morphing_video](https://user-images.githubusercontent.com/25688193/71775924-c056c500-2fcb-11ea-991e-1939d216ce29.gif)<br>

#### ☆ 実行条件２（識別器の BatchNorm なし）

- Epoches : 10<br>
  ![fake_image_epoches10_batchAll](https://user-images.githubusercontent.com/25688193/71560386-91f95700-2aac-11ea-825a-7f2f2b48fc56.png)<br>

- Epoches : 50<br>
  ![fake_image_epoches49_batchAll](https://user-images.githubusercontent.com/25688193/71566490-79fcf400-2afb-11ea-8f4a-ccc7c0f6da45.png)

- Epoches : 1 ~ 50<br>
  ![fake_image_epoches49](https://user-images.githubusercontent.com/25688193/71566489-79fcf400-2afb-11ea-896a-aa4d1e9c5276.gif)<br>

  → 論文にあるように、識別器から BatchNorm を除外すると、生成画像の品質はむしろ悪化している。<br>
  → 論文の意味での BatchNorm の除外になっていない？

- 学習済みモデルからのモーフィング動画<br>
  ![morphing_video](https://user-images.githubusercontent.com/25688193/71775904-8685be80-2fcb-11ea-89f7-deda4e747971.gif)<br>

### ◎ 損失関数のグラフ

#### ☆ 実行条件１（識別器の BatchNorm あり）

- 識別器 : 1 ~ 50 Epoches<br>
  ![image](https://user-images.githubusercontent.com/25688193/71566570-00193a80-2afc-11ea-82b3-a664e83a75b7.png)
  - ピンク色：学習用データの loss 値（ミニバッチ単位）
  - 緑色：テスト用データの loss 値（テスト用データ全体）


- 生成器 : 1 ~ 50 Epoches<br>
  ![image](https://user-images.githubusercontent.com/25688193/71566600-2d65e880-2afc-11ea-83e6-bc36c962c0de.png)
  - ピンク色：学習用データの loss 値（ミニバッチ単位）
  - 緑色：テスト用データの loss 値（テスト用データ全体）

#### ☆ 実行条件２（識別器の BatchNorm なし）

- 識別器 : 1 ~ 50 Epoches<br>
  ![image](https://user-images.githubusercontent.com/25688193/71566527-ba5c7200-2afb-11ea-8d40-d98bf7c8ae73.png)
  - 茶色：学習用データの loss 値（ミニバッチ単位）
  - 水色：テスト用データの loss 値（テスト用データ全体）

- 生成器 : 1 ~ 50 Epoches<br>
  ![image](https://user-images.githubusercontent.com/25688193/71566556-e11aa880-2afb-11ea-9b2e-c6e9855d40f5.png)
  - 茶色：学習用データの loss 値（ミニバッチ単位）
  - 水色：テスト用データの loss 値（テスト用データ全体）

### ◎ 各種オプション引数の設定値

- 実行条件１（識別器の BatchNorm あり）
```python
開始時間： 2019-12-28 14:49:24.849775
PyTorch version : 1.1.0
exper_name: WGAN_train_D_vanilla_Opt_RMSprop_Epoch50_191228_1
device: gpu
dataset: mnist
dataset_dir: ../dataset
results_dir: results
save_checkpoints_dir: checkpoints
load_checkpoints_dir: 
tensorboard_dir: ../tensorboard
n_test: 10000
n_epoches: 50
batch_size: 64
batch_size_test: 256
optimizer: RMSprop
lr_G: 5e-05
lr_D: 5e-05
beta1: 0.5
beta2: 0.999
image_size: 64
n_fmaps: 64
n_input_noize_z: 100
networkD_type: vanilla
n_critic: 5
w_clamp_upper: 0.01
w_clamp_lower: -0.01
n_display_step: 5
n_display_test_step: 100
n_save_step: 10000
seed: 12
debug: True
実行デバイス : cuda
GPU名 : Tesla M60
torch.cuda.current_device() = 0
```

- 実行条件２（識別器の BatchNorm なし）
```python
開始時間： 2019-12-28 14:49:14.457672
PyTorch version : 1.1.0
exper_name: WGAN_train_D_NonBatchNorm_Opt_RMSprop_Epoch50_191228_1
device: gpu
dataset: mnist
dataset_dir: ../dataset
results_dir: results
save_checkpoints_dir: checkpoints
load_checkpoints_dir: 
tensorboard_dir: ../tensorboard
n_test: 10000
n_epoches: 50
batch_size: 64
batch_size_test: 256
optimizer: RMSprop
lr_G: 5e-05
lr_D: 5e-05
beta1: 0.5
beta2: 0.999
image_size: 64
n_fmaps: 64
n_input_noize_z: 100
networkD_type: NonBatchNorm
n_critic: 5
w_clamp_upper: 0.01
w_clamp_lower: -0.01
n_display_step: 5
n_display_test_step: 100
n_save_step: 10000
seed: 12
debug: True
実行デバイス : cuda
GPU名 : Tesla M60
torch.cuda.current_device() = 0
```

## ■ デバッグ情報
