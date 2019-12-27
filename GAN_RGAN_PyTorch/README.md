# GAN_RGAN-GP_PyTorch
RSGAN, RaSGAN, RaLSGAN の PyTorch での実装。

- 参考コード
  - [Github/AlexiaJM/RelativisticGAN](https://github.com/AlexiaJM/RelativisticGAN)

## ■ 項目 [Contents]
1. [動作環境](#動作環境)
1. [使用法](#使用法)
1. [コードの実行結果](#コードの実行結果)
1. [背景理論](https://github.com/Yagami360/MachineLearning-Papers_Survey/issues/51)

## ■ 動作環境

- Mac OS / ubuntu server
- Python : 3.6
- Anaconda : 5.0.1
- PyTorch : 1.1.0

## ■ 使用法

- 学習処理
  ```sh
  # （例１） train RSGAN for MNIST datset using GPU0
  $ python train.py \
    --exper_name RSGAN_train \
    --dataset mnist --image_size 64 \
    --gan_type RSGAN
  ```

  ```sh
  # （例２） train RaSGAN for MNIST datset using GPU0
  $ python train.py \
    --exper_name RaSGAN_train \
    --dataset mnist --image_size 64 \
    --gan_type RaSGAN
  ```

  ```sh
  # （例３） train RaLSGAN for MNIST datset using GPU0
  $ python train.py \
    --exper_name RaLSGAN_train \
    --dataset mnist --image_size 64 \
    --gan_type RaLSGAN
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

#### ☆ RSGAN
- 識別器側<br>
  ![image](https://user-images.githubusercontent.com/25688193/71316691-1e0cec80-24b7-11ea-88f3-ef5b17b6652f.png)<br>
- 生成器側<br>
  ![image](https://user-images.githubusercontent.com/25688193/71316723-3d584980-24b8-11ea-800e-f292ca4624ac.png)<br>
  - 学習データセットの loss 値（）
  - テスト用データセットの loss 値（）
   
#### ☆ RaSGAN
- 識別器側<br>
  <br>

- 生成器側<br>
  <br>
  - 学習データセットの loss 値（オレンジ）
  - テスト用データセットの loss 値（青）

#### ☆ RaLSGAN
- 識別器側<br>
  <br>
- 生成器側<br>
  <br>
  - 学習データセットの loss 値（青）
  - テスト用データセットの loss 値（茶色）

### ◎ 生成器からの生成画像

#### ☆ RSGAN
- Epochs :80<br>
  <br>    
- Epoches : 1 ~ 80<br>
  <br>

#### ☆ RaSGAN
- Epochs :80<br>
  <br>


### ◎ 各種オプション引数の設定値

#### ☆ RSGAN
```python
----------------------------------------------
実行条件
----------------------------------------------
開始時間： 2019-12-27 01:25:17.387356
PyTorch version : 1.1.0
exper_name: RSGAN_train_D_vanilla_Epoch100_191227
device: gpu
dataset: mnist
dataset_dir: ../dataset
results_dir: results
save_checkpoints_dir: checkpoints
load_checkpoints_dir: 
tensorboard_dir: ../tensorboard
n_test: 5000
n_epoches: 100
batch_size: 64
batch_size_test: 256
lr: 0.0001
beta1: 0.5
beta2: 0.999
image_size: 64
n_fmaps: 64
n_input_noize_z: 100
gan_type: RSGAN
networkD_type: vanilla
n_critic: 1
n_display_step: 10
n_display_test_step: 100
n_save_step: 10000
seed: 8
debug: True
実行デバイス : cuda
GPU名 : Tesla M60
torch.cuda.current_device() = 0
```

#### RaSGAN

```python
----------------------------------------------
実行条件
----------------------------------------------
開始時間： 2019-12-27 01:27:35.916840
PyTorch version : 1.1.0
exper_name: RaSGAN_train_D_vanilla_Epoch100_191227
device: gpu
dataset: mnist
dataset_dir: ../dataset
results_dir: results
save_checkpoints_dir: checkpoints
load_checkpoints_dir: 
tensorboard_dir: ../tensorboard
n_test: 5000
n_epoches: 100
batch_size: 64
batch_size_test: 256
lr: 0.0001
beta1: 0.5
beta2: 0.999
image_size: 64
n_fmaps: 64
n_input_noize_z: 100
gan_type: RaSGAN
networkD_type: vanilla
n_critic: 1
n_display_step: 10
n_display_test_step: 100
n_save_step: 10000
seed: 8
debug: True
実行デバイス : cuda
GPU名 : Tesla M60
torch.cuda.current_device() = 0
```

#### ☆ RsLSGAN

```python
----------------------------------------------
実行条件
----------------------------------------------
開始時間： 2019-12-27 01:28:08.538909
PyTorch version : 1.1.0
exper_name: RaLSGAN_train_D_vanilla_Epoch100_191227
device: gpu
dataset: mnist
dataset_dir: ../dataset
results_dir: results
save_checkpoints_dir: checkpoints
load_checkpoints_dir: 
tensorboard_dir: ../tensorboard
n_test: 1000
n_epoches: 100
batch_size: 64
batch_size_test: 256
lr: 0.0001
beta1: 0.5
beta2: 0.999
image_size: 64
n_fmaps: 64
n_input_noize_z: 100
gan_type: RaLSGAN
networkD_type: vanilla
n_critic: 1
n_display_step: 10
n_display_test_step: 100
n_save_step: 10000
seed: 8
debug: True
実行デバイス : cuda
GPU名 : Tesla M60
torch.cuda.current_device() = 0
```

<!--
## ■ デバッグ情報

```python
[Generator]
model_G :
 Generator(
  (layer): Sequential(
    (0): ConvTranspose2d(100, 512, kernel_size=(4, 4), stride=(1, 1), bias=False)
    (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU(inplace)
    (3): ConvTranspose2d(512, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    (4): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (5): ReLU(inplace)
    (6): ConvTranspose2d(256, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    (7): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (8): ReLU(inplace)
    (9): ConvTranspose2d(128, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    (10): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (11): ReLU(inplace)
    (12): ConvTranspose2d(64, 1, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    (13): Tanh()
  )
)
```

```python
model_D :
 Discriminator(
  (layer): Sequential(
    (0): Conv2d(1, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    (1): LeakyReLU(negative_slope=0.2, inplace)
    (2): Conv2d(64, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    (3): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (4): LeakyReLU(negative_slope=0.2, inplace)
    (5): Conv2d(128, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    (6): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (7): LeakyReLU(negative_slope=0.2, inplace)
    (8): Conv2d(256, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    (9): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (10): LeakyReLU(negative_slope=0.2, inplace)
    (11): Conv2d(512, 1, kernel_size=(4, 4), stride=(1, 1), bias=False)
  )
)
```

```python
model_D :
 PatchGANDiscriminator(
  (layer1): Sequential(
    (0): Conv2d(1, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
    (1): LeakyReLU(negative_slope=0.2, inplace)
  )
  (layer2): Sequential(
    (0): Conv2d(64, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
    (1): InstanceNorm2d(128, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
    (2): LeakyReLU(negative_slope=0.2, inplace)
  )
  (layer3): Sequential(
    (0): Conv2d(128, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
    (1): InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
    (2): LeakyReLU(negative_slope=0.2, inplace)
  )
  (layer4): Sequential(
    (0): Conv2d(256, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
    (1): InstanceNorm2d(512, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
    (2): LeakyReLU(negative_slope=0.2, inplace)
  )
  (output_layer): Sequential(
    (0): ZeroPad2d(padding=(1, 0, 1, 0), value=0.0)
    (1): Conv2d(512, 1, kernel_size=(4, 4), stride=(1, 1), padding=(1, 1), bias=False)
  )
)
```

-->