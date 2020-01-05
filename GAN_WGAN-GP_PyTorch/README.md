# GAN_WGAN-GP_PyTorch
WGAN-GPの PyTorch での実装。

- 参考コード
  - [caogang/wgan-gp](https://github.com/caogang/wgan-gp/blob/master/gan_mnist.py)

## ■ 項目 [Contents]
1. [動作環境](#動作環境)
1. [使用法](#使用法)
1. [コードの実行結果](#コードの実行結果)
1. [背景理論](https://github.com/Yagami360/MachineLearning-Papers_Survey/issues/25)

## ■ 動作環境

- Mac OS / ubuntu server
- Python : 3.6
- Anaconda : 5.0.1
- PyTorch : 1.1.0
- tensorboard : 1.13.1
- tensorboardx : 1.9
- tqdm

## ■ 使用法

- 学習処理
  ```sh
  # （例１） train WGAN-GP for MNIST datset using GPU0
  $ python train.py \
    --exper_name WGANGP_train \
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
      --exper_name WGANGP_test_morphing \
      --load_checkpoints_dir checkpoints/WGANGP_train_D_NonBatchNorm_Epoch50_191230 \
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

<!--
#### ☆ 実行条件１（識別器の BatchNorm あり）

- Epoches : 10
  ![fake_image_epoches10_batchAll](https://user-images.githubusercontent.com/25688193/71566616-50909800-2afc-11ea-8b32-98c6f3cfcbef.png)<br>

- Epoches : 50
  ![fake_image_epoches49_batchAll](https://user-images.githubusercontent.com/25688193/71576057-9b2d0700-2b32-11ea-9256-9d9f39528c45.png)<br>

- Epoches : 1 ~ 50<br>
  ![fake_image_epoches49](https://user-images.githubusercontent.com/25688193/71576056-9b2d0700-2b32-11ea-8edd-eb2f210fd707.gif)<br>

  → 生成画像の品質が低い。<br>
  → Epoche数が少ないのが一因か？<br>
  → 識別器の BatchNorm はなしにしたほうが良い？<br>
-->

#### ☆ 実行条件２（識別器の BatchNorm なし）

- Epoches : 10<br>
  ![fake_image_epoches10_batchAll](https://user-images.githubusercontent.com/25688193/71606275-e686fa00-2bb2-11ea-92ba-68b1447af96c.png)<br>
- Epoches : 50<br>
  ![fake_image_epoches49_batchAll](https://user-images.githubusercontent.com/25688193/71776040-b209a880-2fcd-11ea-9cef-0430a0c7b845.png)<br>
- Epoches : 1 ~ 50<br>
  ![fake_image_epoches49](https://user-images.githubusercontent.com/25688193/71618306-6ee0bb80-2c02-11ea-8995-3c677a340fb7.gif)<br>

- 学習済みモデルからのモーフィング動画<br>
  ![morphing_video](https://user-images.githubusercontent.com/25688193/71776028-7f5fb000-2fcd-11ea-89a9-b49c2471f9b6.gif)<br>

### ◎ 損失関数のグラフ

<!--
#### ☆ 実行条件１（識別器の BatchNorm あり）

- 識別器 : 1 ~ 50 Epoches<br>
  ![image](https://user-images.githubusercontent.com/25688193/71576246-5d7cae00-2b33-11ea-922e-3b4cd068a15b.png)

- 生成器 : 1 ~ 50 Epoches<br>
  ![image](https://user-images.githubusercontent.com/25688193/71576272-75ecc880-2b33-11ea-9b10-c6de2c4ab37d.png)

  → loss 値の挙動が不安定で
-->

#### ☆ 実行条件２（識別器の BatchNorm なし）

- 識別器 : 1 ~ 50 Epoches<br>
  ![image](https://user-images.githubusercontent.com/25688193/71618691-1c080380-2c04-11ea-96fd-5ea253718f91.png)<br>
  - ピンク色 : 学習用データセット（ミニバッチ単位）
  - 緑色 : テスト用データセット（データセット全体）

- 生成器 : 1 ~ 50 Epoches<br>
  ![image](https://user-images.githubusercontent.com/25688193/71618705-2c1fe300-2c04-11ea-90e7-a0bff404d066.png)<br>
  - ピンク色 : 学習用データセット（ミニバッチ単位）
  - 緑色 : テスト用データセット（データセット全体）

  ※ テスト用データの gradient penalty loss が０になっているのは、ソフトの不具合
  
### ◎ 各種オプション引数の設定値

<!--
- 実行条件１（識別器の BatchNorm あり）
```python
開始時間： 2019-12-29 01:09:28.921406
PyTorch version : 1.1.0
exper_name: WGANGP_train_D_vanilla_Epoch50_191229
device: gpu
dataset: mnist
dataset_dir: ../dataset
results_dir: results
save_checkpoints_dir: checkpoints
load_checkpoints_dir: 
tensorboard_dir: ../tensorboard
n_test: 5000
n_epoches: 50
batch_size: 64
batch_size_test: 256
lr: 0.0001
beta1: 0.5
beta2: 0.999
image_size: 64
n_fmaps: 64
n_input_noize_z: 100
networkD_type: vanilla
n_critic: 5
lambda_wgangp: 10.0
n_display_step: 5
n_display_test_step: 100
n_save_step: 10000
seed: 8
debug: True
実行デバイス : cuda
GPU名 : Tesla M60
torch.cuda.current_device() = 0
```
-->

- 実行条件２（識別器の BatchNorm なし）

```python
開始時間： 2019-12-30 11:42:27.217281
PyTorch version : 1.1.0
exper_name: WGANGP_train_D_NonBatchNorm_Epoch50_191230
device: gpu
dataset: mnist
dataset_dir: ../dataset
results_dir: results
save_checkpoints_dir: checkpoints
load_checkpoints_dir: 
tensorboard_dir: ../tensorboard
n_test: 5000
n_epoches: 50
batch_size: 64
batch_size_test: 256
lr: 0.0001
beta1: 0.5
beta2: 0.999
image_size: 64
n_fmaps: 64
n_input_noize_z: 100
networkD_type: NonBatchNorm
n_critic: 5
lambda_wgangp: 10.0
n_display_step: 10
n_display_test_step: 100
n_save_step: 10000
seed: 8
debug: True
実行デバイス : cuda
GPU名 : Tesla M60
torch.cuda.current_device() = 0
```

## ■ デバッグ情報

```python
model_D :
 NonBatchNormDiscriminator(
  (layer): Sequential(
    (0): Conv2d(1, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    (1): LeakyReLU(negative_slope=0.2, inplace)
    (2): Conv2d(64, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    (3): LeakyReLU(negative_slope=0.2, inplace)
    (4): Conv2d(128, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    (5): LeakyReLU(negative_slope=0.2, inplace)
    (6): Conv2d(256, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    (7): LeakyReLU(negative_slope=0.2, inplace)
    (8): Conv2d(512, 1, kernel_size=(4, 4), stride=(1, 1), bias=False)
  )
)
```
