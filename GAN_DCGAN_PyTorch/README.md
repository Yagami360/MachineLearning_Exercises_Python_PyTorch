# GAN_DCGAN_PyTorch
DCGAN の PyTorch での実装。

- 参考コード
    - [PyTorch/Tutorials >  DCGAN Tutorial](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html)
    - [PyTorch (12) Generative Adversarial Networks (MNIST) - 人工知能に関する断創録](http://aidiary.hatenablog.com/entry/20180304/1520172429)
    
## ■ 項目 [Contents]
1. [動作環境](#動作環境)
1. [使用法](#使用法)
1. [コードの実行結果](#コードの実行結果)
1. [背景理論](https://github.com/Yagami360/My_NoteBook/blob/master/%E6%83%85%E5%A0%B1%E5%B7%A5%E5%AD%A6/%E6%83%85%E5%A0%B1%E5%B7%A5%E5%AD%A6_%E6%A9%9F%E6%A2%B0%E5%AD%A6%E7%BF%92_%E7%94%9F%E6%88%90%E3%83%A2%E3%83%87%E3%83%AB.md#DCGAN)

## ■ 動作環境

- Mac OS / ubuntu server
- Python : 3.6
- Anaconda : 5.0.1
- PyTorch : 1.1.0

## ■ 使用法

- 学習処理
  ```sh
  # （例１） train DCGAN for MNIST datset using GPU0, using MLP networks only for mnist
  $ python train.py \
    --exper_name DCGAN_MNISTNet_train \
    --dataset mnist --image_size 64 \
    --networkG_type mnist --networkD_type mnist \
    --n_input_noize_z 62
  ```

  ```sh
  # （例２） train DCGAN for MNIST datset using GPU0, using DCGAN networks
  $ python train.py \
    --exper_name DCGAN_train \
    --dataset mnist --image_size 64 \
    --networkG_type vanilla --networkD_type vanilla
  ```

  ```sh
  # （例３） train DCGAN for cifar10 datset using GPU0
  $ python train.py \
    --exper_name DCGAN_train \
    --dataset cifar-10 --image_size 64 \
    --networkG_type vanilla --networkD_type vanilla
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

#### ☆ MLP ネットワーク（実行条件１）
- Epochs :30<br>
  ![fake_image_epoches30_iters1857024_batchAll](https://user-images.githubusercontent.com/25688193/71316862-d6885f80-24ba-11ea-924f-dba470003bc8.png)<br>

#### ☆ DCGAN ネットワークを使用（実行条件２）
- Epoches : 50<br>
  ![fake_image_epoches50_batchAll](https://user-images.githubusercontent.com/25688193/71516740-fde19100-28ed-11ea-8bfc-8f1a0fb6e783.png)<br>
- Epoches : 100<br>
  ![fake_image_epoches99_batchAll](https://user-images.githubusercontent.com/25688193/71537339-fb1e8480-295d-11ea-9b19-2cd25da58d30.png)<br>

### ◎ 損失関数のグラフ

#### ☆ MLP ネットワーク（実行条件１）
- 識別器側<br>
  ![image](https://user-images.githubusercontent.com/25688193/71316820-3b8f8580-24ba-11ea-9153-962dea17b36c.png)<br>
- 生成器側<br>
  ![image](https://user-images.githubusercontent.com/25688193/71316839-80b3b780-24ba-11ea-9a01-1c4ea8039779.png)<br>
  - 学習用データセット（緑）
  - テスト用データセット（灰色）

#### ☆ DCGAN ネットワークを使用（実行条件２）
- 識別器側<br>
  ![image](https://user-images.githubusercontent.com/25688193/71537419-2786d080-295f-11ea-9fed-756536f8f19c.png)
<br>
- 生成器側<br>
  ![image](https://user-images.githubusercontent.com/25688193/71537425-45eccc00-295f-11ea-85ae-687f75430d48.png)<br>

### ◎ 各種オプション引数の設定値

#### ☆ MLP ネットワーク（実行条件１）

```python
----------------------------------------------
実行条件
----------------------------------------------
開始時間： 2019-12-21 12:34:38.011688
PyTorch version : 1.1.0
exper_name: DCGAN_train_G_mnist_D_mnist_Epoch100_191221
device: gpu
dataset: mnist
dataset_dir: ../dataset
results_dir: results
save_checkpoints_dir: checkpoints
load_checkpoints_dir: 
tensorboard_dir: ../tensorboard
n_test: 5000
n_epoches: 100
batch_size: 256
batch_size_test: 256
lr: 0.0001
beta1: 0.5
beta2: 0.999
image_size: 64
n_fmaps: 64
n_input_noize_z: 62
networkG_type: mnist
networkD_type: mnist
n_display_step: 100
n_display_test_step: 1000
n_save_step: 10000
seed: 8
debug: True
実行デバイス : cuda
GPU名 : Tesla M60
torch.cuda.current_device() = 0
```

#### ☆ DCGAN ネットワークを使用（実行条件２）

```python
----------------------------------------------
実行条件
----------------------------------------------
開始時間： 2019-12-27 01:24:01.742542
PyTorch version : 1.1.0
exper_name: DCGAN_train_G_vanilla_D_vanilla_Epoch100_191227
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
networkG_type: vanilla
networkD_type: vanilla
n_display_step: 50
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
[Generator for MNIST]
model_G :
 MNISTGenerator(
  (fc_layer): Sequential(
    (0): Linear(in_features=100, out_features=1024, bias=True)
    (1): BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU()
    (3): Linear(in_features=1024, out_features=6272, bias=True)
    (4): BatchNorm1d(6272, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (5): ReLU()
  )
  (deconv_layer): Sequential(
    (0): ConvTranspose2d(128, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
    (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU()
    (3): ConvTranspose2d(64, 1, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
    (4): Sigmoid()
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

```python
model_D :
 MNISTDiscriminator(
  (conv_layer): Sequential(
    (0): Conv2d(1, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
    (1): LeakyReLU(negative_slope=0.2, inplace)
    (2): Conv2d(64, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
    (3): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (4): LeakyReLU(negative_slope=0.2, inplace)
  )
  (fc_layer): Sequential(
    (0): Linear(in_features=6272, out_features=1024, bias=True)
    (1): BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): LeakyReLU(negative_slope=0.2)
    (3): Linear(in_features=1024, out_features=1, bias=True)
    (4): Sigmoid()
  )
)
```