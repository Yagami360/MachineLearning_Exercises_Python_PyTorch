# GAN_DCGAN_PyTorch
DCGAN の PyTorch での実装。

## ■ 項目 [Contents]
1. [動作環境](#動作環境)
1. [使用法](#使用法)
1. [コードの実行結果](#コードの実行結果)
1. [背景理論](https://github.com/Yagami360/My_NoteBook/blob/master/%E6%83%85%E5%A0%B1%E5%B7%A5%E5%AD%A6/%E6%83%85%E5%A0%B1%E5%B7%A5%E5%AD%A6_%E6%A9%9F%E6%A2%B0%E5%AD%A6%E7%BF%92_%E7%94%9F%E6%88%90%E3%83%A2%E3%83%87%E3%83%AB.md#DCGAN)

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
[main_mnist.py]
#DEVICE = "CPU"               # 使用デバイス ("CPU" or "GPU")
DEVICE = "GPU"                # 使用デバイス ("CPU" or "GPU")
DATASET_PATH = "./dataset"    # 学習用データセットへのパス
NUM_SAVE_STEP = 1             # 自動生成画像の保存間隔（エポック単位）

NUM_EPOCHES = 10              # エポック数（学習回数）
LEARNING_RATE = 0.0002        # 学習率
NUM_FEATURE_MAPS = 64         # 特徴マップの枚数
BATCH_SIZE = 128              # ミニバッチサイズ
NUM_INPUT_NOIZE_Z = 62        # 生成器に入力するノイズ z の次数
```


```python
[main.py]
#DEVICE = "CPU"               # 使用デバイス ("CPU" or "GPU")
DEVICE = "GPU"                # 使用デバイス ("CPU" or "GPU")
DATASET = "MNIST"            # データセットの種類（"MNIST" or "CIFAR-10"）
#DATASET = "CIFAR-10"          # データセットの種類（"MNIST" or "CIFAR-10"）
DATASET_PATH = "./dataset"    # 学習用データセットへのパス
NUM_SAVE_STEP = 1             # 自動生成画像の保存間隔（エポック単位）

NUM_EPOCHES = 50              # エポック数（学習回数）
LEARNING_RATE = 0.00005       # 学習率
IMAGE_SIZE = 64               # 入力画像のサイズ（pixel単位）
NUM_CHANNELS = 1              # 入力画像のチャンネル数
NUM_FEATURE_MAPS = 64         # 特徴マップの枚数
BATCH_SIZE = 128              # ミニバッチサイズ
NUM_INPUT_NOIZE_Z = 100       # 生成器に入力するノイズ z の次数
```


<a id="コードの実行結果"></a>

## ■ コードの実行結果：`main_mnist.py`

|パラメータ名|値（実行条件１）|
|---|---|
|使用デバイス：`DEVICE`|GPU|
|シード値|`random.seed(8)`<br>`np.random.seed(8)`<br>`torch.manual_seed(8)`|
|エポック数：`NUM_EPOCHES`|50|
|バッチサイズ：`BATCH_SIZE`|128|
|最適化アルゴリズム|Adam|
|学習率：`LEARNING_RATE`|0.0002|
|減衰率 beta1|0.5|
|生成器に入力するノイズ z の次数：`NUM_INPUT_NOIZE_Z`|62|
|特徴マップの枚数：`NUM_FEATURE_MAPS`|64|

### ◎ 損失関数のグラフ（実行条件１）：`main_mnist.py`
![DCGANforMNIST_Loss_epoches10_lr0 0002_batchsize128](https://user-images.githubusercontent.com/25688193/56814818-eb084f80-687a-11e9-967f-062388b8d90a.png)<br>

### ◎ 生成器から生成された自動生成画像（実行条件１）：`main_mnist.py`

- エポック数 : 1 / イテレーション回数：468<br>
![DCGANforMNIST_Image_epoches0_iters468](https://user-images.githubusercontent.com/25688193/56814485-4423b380-687a-11e9-932b-081b1e56f9e7.png)<br>

- エポック数 : 2 / イテレーション回数：936<br>
![DCGANforMNIST_Image_epoches1_iters936](https://user-images.githubusercontent.com/25688193/56814486-4423b380-687a-11e9-9c42-4dfa1177624f.png)<br>

- エポック数 : 3 / イテレーション回数：1404<br>
![DCGANforMNIST_Image_epoches2_iters1404](https://user-images.githubusercontent.com/25688193/56814487-4423b380-687a-11e9-88b4-e22648dd4f49.png)<br>

- エポック数 : 4 / イテレーション回数：1872<br>
![DCGANforMNIST_Image_epoches3_iters1872](https://user-images.githubusercontent.com/25688193/56814488-44bc4a00-687a-11e9-80cb-fea51477b1ab.png)<br>

- エポック数 : 5 / イテレーション回数 : 2340<br>
![DCGANforMNIST_Image_epoches4_iters2340](https://user-images.githubusercontent.com/25688193/56814547-674e6300-687a-11e9-9edd-1654bb2ec6d1.png)<br>

- エポック数 : 6 / イテレーション回数 : 2808<br>
![DCGANforMNIST_Image_epoches5_iters2808](https://user-images.githubusercontent.com/25688193/56814608-84833180-687a-11e9-9572-a69267462c7c.png)<br>

- エポック数 : 7 / イテレーション回数 : 3276<br>
![DCGANforMNIST_Image_epoches6_iters3276](https://user-images.githubusercontent.com/25688193/56814654-9a90f200-687a-11e9-8218-fb47b792f621.png)<br>

- エポック数 : 8 / イテレーション回数 : 3744<br>
![DCGANforMNIST_Image_epoches7_iters3744](https://user-images.githubusercontent.com/25688193/56814693-ad0b2b80-687a-11e9-9eca-f55936cb5bc8.png)<br>

- エポック数 : 9 / イテレーション回数 : 4212<br>
![DCGANforMNIST_Image_epoches8_iters4212](https://user-images.githubusercontent.com/25688193/56814780-d1ff9e80-687a-11e9-8521-8315f9e0b370.png)<br>

- エポック数 : 10 / イテレーション回数 : 4680<br>
![DCGANforMNIST_Image_epoches9_iters4680](https://user-images.githubusercontent.com/25688193/56814816-eb084f80-687a-11e9-94a9-4cf7140faaed.png)<br>


## ■ コードの実行結果：`main.py`

|パラメータ名|値（実行条件１）|値（実行条件２）|
|---|---|---|
|学習用データセット：`DATASET`|"MNIST"|"CIFAR-10"|
|使用デバイス：`DEVICE`|GPU|←|
|シード値|`random.seed(8)`<br>`np.random.seed(8)`<br>`torch.manual_seed(8)`|←|
|エポック数：`NUM_EPOCHES`|10|←|
|バッチサイズ：`BATCH_SIZE`|128|←|
|最適化アルゴリズム|Adam|←|
|学習率：`LEARNING_RATE`|0.00005|←|
|減衰率 beta1|0.5|←|
|生成器に入力するノイズ z の次数：`NUM_INPUT_NOIZE_Z`|100|100|
|入力画像のサイズ：`IMAGE_SIZE`|64|64|
|入力画像のチャンネル数：`NUM_CHANNELS`|1|3|
|特徴マップの枚数：`NUM_FEATURE_MAPS`|64|64|

### ◎ 損失関数のグラフ（実行条件１）
<br>

### ◎ 生成器から生成された自動生成画像（実行条件１）

- エポック数 : 1 / イテレーション回数：468<br>
![DCGAN_Image_epoches0_iters468](https://user-images.githubusercontent.com/25688193/56815729-02e0d300-687d-11e9-8dea-a2bdce4ae4b9.png)<br>

- エポック数 : 2 / イテレーション回数：936<br>
![DCGAN_Image_epoches1_iters936](https://user-images.githubusercontent.com/25688193/56815628-cdd48080-687c-11e9-973c-bc0e6d188034.png)<br>

- エポック数 : 3 / イテレーション回数：1404<br>
<br>

- エポック数 : 4 / イテレーション回数：1872<br>
<br>

- エポック数 : 5 / イテレーション回数 : 2340<br>
<br>

- エポック数 : 6 / イテレーション回数 : 2808<br>
<br>

- エポック数 : 7 / イテレーション回数 : 3276<br>
<br>

- エポック数 : 8 / イテレーション回数 : 3744<br>
<br>

- エポック数 : 9 / イテレーション回数 : 4212<br>
<br>

- エポック数 : 10 / イテレーション回数 : 4680<br>
<br>
