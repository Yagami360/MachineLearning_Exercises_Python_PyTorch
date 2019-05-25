# ResNet_PyTorch
ResNet-18 の PyTorch での実装<br>

- 参考コード
    - [PyTorch ResNet: Building, Training and Scaling Residual Networks on PyTorch](https://missinglink.ai/guides/deep-learning-frameworks/pytorch-resnet-building-training-scaling-residual-networks-pytorch/)
    - [ResNet PyTorch](http://www.pabloruizruiz10.com/resources/CNNs/ResNet-PyTorch.html)


## ■ 項目 [Contents]
1. [動作環境](#動作環境)
1. [使用法](#使用法)
1. [コードの実行結果](#コードの実行結果)
1. [背景理論](http://yagami12.hatenablog.com/entry/2017/09/17/111935#ResNet%EF%BC%88%E6%AE%8B%E5%B7%AE%E3%83%8D%E3%83%83%E3%83%88%E3%83%AF%E3%83%BC%E3%82%AF%EF%BC%89)

## ■ 動作環境

- Windows 10
- Geforce GTX1050 / VRAM:2GB
- Python : 3.6
- Anaconda : 5.0.1
- PyTorch : 1.1.0

## ■ 使用法

- 使用法

```
$ python main.py
```

- CIFAR-10 の場合（学習時）

```
$ python main.py --device GPU --run_mode train --dataset CIFAR-10 --n_epoches 10  --batch_size 32 --learning_rate 0.001
```

- CIFAR-10 の場合（推論時）

```
$ python main.py --device GPU --run_mode test --dataset CIFAR-10
```


- 設定可能なコマンドライン引数

|引数名|意味|値 (Default値)|
|---|---|---|
|`--device`|実行デバイス|`GPU` (Default) or `CPU`|
|`--run_mode`|実行デバイス|`train` (Default) or `add_train` or `test`|
|`--dataset`|データセット|`MNIST` (Default) or `CIFAR-10`|
|`--dataset_path`|データセットの保存先|`./dataset` (Default)|
|`--image_size`|データセットの画像サイズ（pixel単位）|`224` (Default)<br>ImageNetのサイズ|
|`--n_classes`|分類クラス数|`10` (Default)|
|`--n_epoches`|エポック数|`50` (Default)|
|`--batch_size`|バッチサイズ|`32` (Default)|
|`--learning_rate`|学習率|`0.001` (Default)|
|`--result_path`|学習結果のフォルダ|`./result` (Default)|
|`--xxx`|xxx|`xxx` (Default)|


<a id="コードの実行結果"></a>

## ■ コードの実行結果：`main.py`

|パラメータ名|値（実行条件１）|値（実行条件２）|
|---|---|---|
|使用デバイス：`--device`|GPU|←|
|データセット：`--dataset`|CIFAR-10|
|画像サイズ：`--image_size`|224|
|分類クラス数：`--n_classes`|10|
|エポック数：`--n_epoches`|5||
|バッチサイズ：`--batch_size`|32|←|
|最適化アルゴリズム|Adam|←|
|学習率：`--learning_rate`|0.001|←|
|減衰率 beta1|0.5|←|
|シード値|`random.seed(8)`<br>`np.random.seed(8)`<br>`torch.manual_seed(8)`|←|

- ネットワーク構成（実行条件１）
```python
_model : ResNet18(
  (_layer0): Sequential(
    (0): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))
    (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): LeakyReLU(negative_slope=0.2, inplace)
    (3): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
  )
  (_layer1): Sequential(
    (0): BasicBlock(
      (_layer1): Sequential(
        (0): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): LeakyReLU(negative_slope=0.2, inplace)
      )
      (_layer2): Sequential(
        (0): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (_shortcut_connections): Sequential()
    )
    (1): BasicBlock(
      (_layer1): Sequential(
        (0): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): LeakyReLU(negative_slope=0.2, inplace)
      )
      (_layer2): Sequential(
        (0): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (_shortcut_connections): Sequential()
    )
  )
  (_layer2): Sequential(
    (0): BasicBlock(
      (_layer1): Sequential(
        (0): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): LeakyReLU(negative_slope=0.2, inplace)
      )
      (_layer2): Sequential(
        (0): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (_shortcut_connections): Sequential(
        (0): Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)
        (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (1): BasicBlock(
      (_layer1): Sequential(
        (0): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): LeakyReLU(negative_slope=0.2, inplace)
      )
      (_layer2): Sequential(
        (0): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (_shortcut_connections): Sequential()
    )
  )
  (_layer3): Sequential(
    (0): BasicBlock(
      (_layer1): Sequential(
        (0): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): LeakyReLU(negative_slope=0.2, inplace)
      )
      (_layer2): Sequential(
        (0): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (_shortcut_connections): Sequential(
        (0): Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2), bias=False)
        (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (1): BasicBlock(
      (_layer1): Sequential(
        (0): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): LeakyReLU(negative_slope=0.2, inplace)
      )
      (_layer2): Sequential(
        (0): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (_shortcut_connections): Sequential()
    )
  )
  (_layer4): Sequential(
    (0): BasicBlock(
      (_layer1): Sequential(
        (0): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): LeakyReLU(negative_slope=0.2, inplace)
      )
      (_layer2): Sequential(
        (0): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (_shortcut_connections): Sequential(
        (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
        (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (1): BasicBlock(
      (_layer1): Sequential(
        (0): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): LeakyReLU(negative_slope=0.2, inplace)
      )
      (_layer2): Sequential(
        (0): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (_shortcut_connections): Sequential()
    )
  )
  (_avgpool): AvgPool2d(kernel_size=7, stride=1, padding=0)
  (_fc_layer): Linear(in_features=512, out_features=10, bias=True)
)
```

### ◎ 損失関数のグラフ（実行条件１）
![ResNet18_Loss_epoches5_lr0 001_batchsize32](https://user-images.githubusercontent.com/25688193/58368136-bf34c400-7f23-11e9-8ca1-9880de75776b.png)

### ◎ 正解率

- エポック : 5 / Acuraccy [test data] : 0.738
- エポック : 10 / Acuraccy [test data] : 0.80390

<!--
|ラベル|Acuraccy [test data]|サンプル数|
|---|---|---|
|全ラベルでの平均|0.738(±)|10,000 個|
|0 : airplane|xxx|1000 個|
|1 : automoblie|xxx|1000 個|
|2 : bird|xxx|1000 個|
|3 : cat|xxx|1000 個|
|4 : deer|xxx|1000 個|
|5 : dog|xxx|1000 個|
|6 : frog|xxx|1000 個|
|7 : horse|xxx|1000 個|
|8 : ship|xxx|1000 個|
|9 : tuck|xxx|1000 個|
-->
