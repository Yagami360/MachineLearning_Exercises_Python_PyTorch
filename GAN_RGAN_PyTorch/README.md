# GAN_RGAN_PyTorch
Relativistic GANs (RSGAN, RaSGAN, RaLSGAN) の PyTorch での実装。

- 参考コード
  - [Github/AlexiaJM/RelativisticGAN](https://github.com/AlexiaJM/RelativisticGAN)

## ■ 項目 [Contents]
1. [動作環境](#動作環境)
1. [使用法](#使用法)
1. [コードの実行結果](#コードの実行結果)
1. [背景理論](https://github.com/Yagami360/MachineLearning-Papers_Survey/issues/51)

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

- 推論処理（学習済みモデルから画像生成）
  ```sh
  $ python test.py --load_checkpoints_dir ${LOAD_CHAECKPOINTS_DIR}
  ```

- 推論処理（学習済みモデルからモーフィング動画の作成）
  ```sh
  # モーフィング動画の作成
  $ python test_morphing.py --load_checkpoints_dir ${LOAD_CHAECKPOINTS_DIR}
  ```
  ```sh
  # （例）
  $ python test_morphing.py \
      --exper_name RGAN_test_morphing \
      --load_checkpoints_dir checkpoints/RaLSGAN_train_D_vanilla_Epoch100_191227 \
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

#### ☆ RSGAN

- Epochs : 10<br>
  ![fake_image_epoches10_batchAll](https://user-images.githubusercontent.com/25688193/71560073-873cc300-2aa8-11ea-86e7-cb686d2b31ea.png)

- Epochs : 50<br>
  ![fake_image_epoches50_batchAll](https://user-images.githubusercontent.com/25688193/71560088-b05d5380-2aa8-11ea-83bd-f2d6d46b500f.png)

- Epochs : 100<br>
  ![fake_image_epoches96_batchAll](https://user-images.githubusercontent.com/25688193/71560109-ee5a7780-2aa8-11ea-876a-f88dd4aeb80f.png)<br>
  → 他の RaSGAN, RsLSGAN と比較して、生成画像の品質は低い。

- Epoches : 1 ~ 100<br>
  ![fake_image_epoches96](https://user-images.githubusercontent.com/25688193/71560087-b05d5380-2aa8-11ea-9a39-be6a35e76826.gif)<br>
  → 入力ノイズ z を固定した場合でも、学習中の生成画像が安定しない。

- 学習済みモデルからのモーフィング動画<br>
  ![morphing_video](https://user-images.githubusercontent.com/25688193/71775768-dc596700-2fc9-11ea-94d9-1f2580f2eeb5.gif)<br>

#### ☆ RaSGAN

- Epochs :10<br>
  ![fake_image_epoches10_batchAll](https://user-images.githubusercontent.com/25688193/71560121-1ea21600-2aa9-11ea-8e34-5dd0e1119087.png)

- Epochs :50<br>
  ![fake_image_epoches50_batchAll](https://user-images.githubusercontent.com/25688193/71560120-1ea21600-2aa9-11ea-8f12-86aa5d2605c9.png)

- Epochs :100<br>
  ![fake_image_epoches99_batchAll](https://user-images.githubusercontent.com/25688193/71566312-f8f12d00-2af9-11ea-85d7-0823a6ec557b.png)
<br>

  → vanilla GAN や WGAN, WGAN-GP や RGAN と比較して、生成画像の品質が高い。<br>
  → 又、WGAN, WGAN-GP や RGAN と比較して、学習の初期段階から品質の高い画像が生成出来ている。

- Epoches : 1 ~100<br>
  ![fake_image_epoches99](https://user-images.githubusercontent.com/25688193/71566310-f8f12d00-2af9-11ea-9b9d-52bef0b9a4e8.gif)

- 学習済みモデルからのモーフィング動画<br>
  ![morphing_video](https://user-images.githubusercontent.com/25688193/71775797-32c6a580-2fca-11ea-8837-1ee9973c451c.gif)<br>

#### ☆ RaLSGAN

- Epochs :10<br>
  ![fake_image_epoches10_batchAll](https://user-images.githubusercontent.com/25688193/71560019-e0582700-2aa7-11ea-916d-110007b85270.png)

- Epochs :50<br>
  ![fake_image_epoches50_batchAll](https://user-images.githubusercontent.com/25688193/71560018-e0582700-2aa7-11ea-9774-fb647771160f.png)

- Epochs :100<br>
  ![fake_image_epoches99_batchAll](https://user-images.githubusercontent.com/25688193/71566336-2047fa00-2afa-11ea-848d-839d2609e6a0.png)<br>
  → vanilla GAN や WGAN, WGAN-GP や RGAN と比較して、生成画像の品質が高い。<br>
  → 又、WGAN, WGAN-GP や RGAN と比較して、学習の初期段階から品質の高い画像が生成出来ている。

- Epoches : 1 ~100<br>
  ![fake_image_epoches99](https://user-images.githubusercontent.com/25688193/71566335-2047fa00-2afa-11ea-801d-cea2da92521e.gif)<br>

- 学習済みモデルからのモーフィング動画<br>
  ![morphing_video](https://user-images.githubusercontent.com/25688193/71775733-21c96480-2fc9-11ea-91f2-47ea084c40b1.gif)<br>

### ◎ 損失関数のグラフ

#### ☆ RSGAN
- 識別器側 : Epoches 1~100<br>
  ![image](https://user-images.githubusercontent.com/25688193/71543060-cdfbc180-29b1-11ea-9267-7789725bebee.png)<br>
- 生成器側 : Epoches 1~100<br>
  ![image](https://user-images.githubusercontent.com/25688193/71543057-c0ded280-29b1-11ea-94a7-c8db7a41a3f0.png)<br>
   
#### ☆ RaSGAN
- 識別器側<br>
  ![image](https://user-images.githubusercontent.com/25688193/71566378-846abe00-2afa-11ea-9a7f-f30a42d4b0d4.png)<br>

- 生成器側<br>
  ![image](https://user-images.githubusercontent.com/25688193/71566389-99475180-2afa-11ea-90bd-21df297624f7.png)<br>

#### ☆ RaLSGAN
- 識別器側<br>
  ![image](https://user-images.githubusercontent.com/25688193/71566354-52f1f280-2afa-11ea-8f15-a01034f2a78f.png)<br>
- 生成器側<br>
  ![image](https://user-images.githubusercontent.com/25688193/71566365-64d39580-2afa-11ea-8a35-ca3f34deef0d.png)<br>

### ◎ 各種オプション引数の設定値

#### ☆ RSGAN
```python
開始時間： 2019-12-27 15:21:10.851957
PyTorch version : 1.1.0
exper_name: RSGAN_train_D_vanilla_Epoch100_191227_1
device: gpu
dataset: mnist
dataset_dir: ../dataset
results_dir: results
save_checkpoints_dir: checkpoints
load_checkpoints_dir: 
tensorboard_dir: ../tensorboard
n_test: 10000
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
開始時間： 2019-12-29 01:35:41.331311
PyTorch version : 1.1.0
exper_name: RaSGAN_train_D_vanilla_Epoch100_191229
device: gpu
dataset: mnist
dataset_dir: ../dataset
results_dir: results
save_checkpoints_dir: checkpoints
load_checkpoints_dir: 
tensorboard_dir: ../tensorboard
n_test: 10000
n_epoches: 100
batch_size: 64
batch_size_test: 64
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
開始時間： 2019-12-28 11:40:47.254753
PyTorch version : 1.1.0
exper_name: RaLSGAN_train_D_vanilla_Epoch100_191228
device: gpu
dataset: mnist
dataset_dir: ../dataset
results_dir: results
save_checkpoints_dir: checkpoints
load_checkpoints_dir: 
tensorboard_dir: ../tensorboard
n_test: 10000
n_epoches: 100
batch_size: 64
batch_size_test: 64
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