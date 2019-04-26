# GAN_CGAN_PyTorch【実装中...】
Conditional GAN（CGAN）の PyTorch での実装。<br>

ネットワーク構成は、CNN を使用（DCGANベース）

## ■ 項目 [Contents]
1. [動作環境](#動作環境)
1. [使用法](#使用法)
1. [コード説明＆実行結果](#コード説明＆実行結果)
    1. `main.py`
    1. `main_mnist.py`
1. [背景理論](https://github.com/Yagami360/My_NoteBook/blob/master/%E6%83%85%E5%A0%B1%E5%B7%A5%E5%AD%A6/%E6%83%85%E5%A0%B1%E5%B7%A5%E5%AD%A6_%E6%A9%9F%E6%A2%B0%E5%AD%A6%E7%BF%92_%E7%94%9F%E6%88%90%E3%83%A2%E3%83%87%E3%83%AB.md#ConditionalGAN%EF%BC%88CGAN%EF%BC%89)

## ■ 動作環境

- Windows 10
- Geforce GTX1050 / VRAM:2GB
- Python : 3.6
- Anaconda : 5.0.1
- PyTorch : 1.0.1

## ■ 使用法

- 使用法
```
$ python main.py
```

- 設定可能な定数
```python
[main.py]
#DEVICE = "CPU"               # 使用デバイス ("CPU" or "GPU")
DEVICE = "GPU"                # 使用デバイス ("CPU" or "GPU")
DATASET = "MNIST"             # データセットの種類（"MNIST" or "CIFAR-10"）
DATASET_PATH = "./dataset"    # 学習用データセットへのパス
NUM_SAVE_STEP = 1             # 自動生成画像の保存間隔（エポック単位）

NUM_EPOCHES = 10              # エポック数（学習回数）
LEARNING_RATE = 0.0002        # 学習率
BATCH_SIZE = 128              # ミニバッチサイズ
IMAGE_SIZE = 64               # 入力画像のサイズ（pixel単位）
NUM_CHANNELS = 1              # 入力画像のチャンネル数
NUM_FEATURE_MAPS = 64         # 特徴マップの枚数
NUM_INPUT_NOIZE_Z = 62        # 生成器に入力するノイズ z の次数
NUM_CLASSES = 10              # クラスラベル y の次元数
```


<a id="コード説明＆実行結果"></a>

## ■ コード説明＆実行結果

### ◎ コードの実行結果：`main.py`

|パラメータ名|値（実行条件１）|値（実行条件２）|
|---|---|---|
|学習用データセット：`DATASET`|"MNIST"|"CIFAR-10"|
|使用デバイス：`DEVICE`|GPU|←|
|シード値|`random.seed(8)`<br>`np.random.seed(8)`<br>`torch.manual_seed(8)`|←|
|エポック数：`NUM_EPOCHES`|10|←|
|バッチサイズ：`BATCH_SIZE`|128|←|
|最適化アルゴリズム|Adam|←|
|学習率：`LEARNING_RATE`|0.0002|←|
|減衰率 beta1|0.5|←|
|生成器に入力するノイズ z の次数：`NUM_INPUT_NOIZE_Z`|100|100|
|入力画像のサイズ：`IMAGE_SIZE`|64|64|
|入力画像のチャンネル数：`NUM_CHANNELS`|1|3|
|特徴マップの枚数：`NUM_FEATURE_MAPS`|64|64|
|クラスラベルの個数：`NUM_CLASSES`|10|x|


### ◎ コードの実行結果：`main_mnist.py`

|パラメータ名|値（実行条件１）|
|---|---|
|学習用データセット：`DATASET`|"MNIST"|
|使用デバイス：`DEVICE`|GPU|
|シード値|`random.seed(8)`<br>`np.random.seed(8)`<br>`torch.manual_seed(8)`|
|エポック数：`NUM_EPOCHES`|10|
|バッチサイズ：`BATCH_SIZE`|128|
|最適化アルゴリズム|Adam|
|学習率：`LEARNING_RATE`|0.0002|
|減衰率 beta1|0.5|
|生成器に入力するノイズ z の次数：`NUM_INPUT_NOIZE_Z`|62|
|特徴マップの枚数：`NUM_FEATURE_MAPS`|64|


#### ☆ 損失関数のグラフ（実行条件１）
<br>

#### ☆ 生成器から生成された自動生成画像（実行条件１）

- エポック数 : 1 / イテレーション回数：468<br>
<br>

- エポック数 : 2 / イテレーション回数：936<br>
<br>

- エポック数 : 3 / イテレーション回数：1404<br>


- エポック数 : 4 / イテレーション回数：1872<br>


- エポック数 : 5 / イテレーション回数 : 2340<br>


- エポック数 : 10 / イテレーション回数 : 4680<br>

