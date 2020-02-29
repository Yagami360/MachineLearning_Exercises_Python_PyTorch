# ResNet_PyTorch
ResNet-18 の PyTorch での実装

- 参考コード
    - [PyTorch ResNet: Building, Training and Scaling Residual Networks on PyTorch](https://missinglink.ai/guides/deep-learning-frameworks/pytorch-resnet-building-training-scaling-residual-networks-pytorch/)
    - [ResNet PyTorch](http://www.pabloruizruiz10.com/resources/CNNs/ResNet-PyTorch.html)


## ■ 項目 [Contents]
1. [動作環境](#動作環境)
1. [使用法](#使用法)
1. [コードの実行結果](#コードの実行結果)
1. [背景理論](http://yagami12.hatenablog.com/entry/2017/09/17/111935#ResNet%EF%BC%88%E6%AE%8B%E5%B7%AE%E3%83%8D%E3%83%83%E3%83%88%E3%83%AF%E3%83%BC%E3%82%AF%EF%BC%89)

## ■ 使用法

- 学習処理
  ```sh
  # （例１） train ResNet18 for MNIST datset using GPU0
  $ python train.py \
    --exper_name ResNet18_train_mnist \
    --dataset mnist --image_size 224 --n_classes 10
  ```

  ```sh
  # （例２） train ResNet18 for CIFAR-10 datset using GPU0
  $ python train.py \
    --exper_name ResNet_train_cifar10 \
    --dataset cifar-10 --image_size 224 --n_classes 10
  ```

- 推論処理
  ```sh
  # （例１） test ResNet18 for MNIST datset using GPU0
  $ python test.py \
    --exper_name ResNet18_test_mnist \
    --load_checkpoints_dir ${LOAD_CHECKPOINTS_DIR} \
    --dataset mnist --image_size 224 --n_classes 10
  ```

  ```sh
  # （例1-1） test ResNet18 for MNIST datset using GPU0
  $ python test.py \
    --exper_name ResNet18_test_mnist \
    --dataset_dir ../dataset \
    --load_checkpoints_dir checkpoints/ResNet18_train_Epoch10_191230 \
    --dataset mnist --image_size 224 --n_classes 10
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

### ◎ 正解率のグラフ

- MNIST / Epoches 1~10<br>
  ![image](https://user-images.githubusercontent.com/25688193/71580241-257d6700-2b43-11ea-9121-0db9b1ecfecd.png)<br>
  - ピンク：学習用データセット（データセット全体）
  - 緑：テスト用データセット（データセット全体）

### ◎ 損失関数のグラフ

- MNIST / Epoches 1~10<br>
  ![image](https://user-images.githubusercontent.com/25688193/71580281-4a71da00-2b43-11ea-9efe-e1527e5f9933.png)<br>
  - ピンク：学習用データセット（ミニバッチ単位）
  - 緑：テスト用データセット（データセット全体）

### ◎ 各種オプション引数の設定値

```python
開始時間： 2019-12-30 03:06:20.420439
PyTorch version : 1.1.0
exper_name: ResNet18_train_Epoch10_191230
device: gpu
dataset: mnist
dataset_dir: ../dataset
results_dir: results
save_checkpoints_dir: checkpoints
load_checkpoints_dir: 
tensorboard_dir: ../tensorboard
n_test: 10000
n_epoches: 10
batch_size: 32
batch_size_test: 256
lr: 0.001
beta1: 0.5
beta2: 0.999
image_size: 224
n_fmaps: 64
n_classes: 10
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
model :
 ResNet18(
  (layer0): Sequential(
    (0): Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))
    (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): LeakyReLU(negative_slope=0.2, inplace)
    (3): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
  )
  (layer1): Sequential(
    (0): BasicBlock(
      (layer1): Sequential(
        (0): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): LeakyReLU(negative_slope=0.2, inplace)
      )
      (layer2): Sequential(
        (0): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (shortcut_connections): Sequential()
    )
    (1): BasicBlock(
      (layer1): Sequential(
        (0): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): LeakyReLU(negative_slope=0.2, inplace)
      )
      (layer2): Sequential(
        (0): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (shortcut_connections): Sequential()
    )
  )
  (layer2): Sequential(
    (0): BasicBlock(
      (layer1): Sequential(
        (0): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): LeakyReLU(negative_slope=0.2, inplace)
      )
      (layer2): Sequential(
        (0): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (shortcut_connections): Sequential(
        (0): Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)
        (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (1): BasicBlock(
      (layer1): Sequential(
        (0): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): LeakyReLU(negative_slope=0.2, inplace)
      )
      (layer2): Sequential(
        (0): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (shortcut_connections): Sequential()
    )
  )
  (layer3): Sequential(
    (0): BasicBlock(
      (layer1): Sequential(
        (0): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): LeakyReLU(negative_slope=0.2, inplace)
      )
      (layer2): Sequential(
        (0): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (shortcut_connections): Sequential(
        (0): Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2), bias=False)
        (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (1): BasicBlock(
      (layer1): Sequential(
        (0): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): LeakyReLU(negative_slope=0.2, inplace)
      )
      (layer2): Sequential(
        (0): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (shortcut_connections): Sequential()
    )
  )
  (layer4): Sequential(
    (0): BasicBlock(
      (layer1): Sequential(
        (0): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): LeakyReLU(negative_slope=0.2, inplace)
      )
      (layer2): Sequential(
        (0): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (shortcut_connections): Sequential(
        (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
        (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (1): BasicBlock(
      (layer1): Sequential(
        (0): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): LeakyReLU(negative_slope=0.2, inplace)
      )
      (layer2): Sequential(
        (0): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (shortcut_connections): Sequential()
    )
  )
  (avgpool): AvgPool2d(kernel_size=7, stride=1, padding=0)
  (fc_layer): Linear(in_features=512, out_features=10, bias=True)
)
```

```python

```
