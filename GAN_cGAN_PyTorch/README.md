# GAN_cGAN_PyTorch
Conditional GAN（cGAN）の PyTorch での実装。<br>

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
DATASET = "MNIST"             # データセットの種類（"MNIST" or "CIFAR-10"）
DATASET_PATH = "./dataset"    # 学習用データセットへのパス
NUM_SAVE_STEP = 1             # 自動生成画像の保存間隔（エポック単位）

NUM_EPOCHES = 50              # エポック数（学習回数）
LEARNING_RATE = 0.00005       # 学習率
BATCH_SIZE = 128              # ミニバッチサイズ
IMAGE_SIZE = 64               # 入力画像のサイズ（pixel単位）
NUM_CHANNELS = 1              # 入力画像のチャンネル数
NUM_FEATURE_MAPS = 64         # 特徴マップの枚数
NUM_INPUT_NOIZE_Z = 100       # 生成器に入力するノイズ z の次数
NUM_CLASSES = 10              # クラスラベル y の次元数
```


<a id="コード説明＆実行結果"></a>

## ■ コード説明＆実行結果

### ◎ コードの実行結果：`main_mnist.py`

|パラメータ名|値（実行条件１）|
|---|---|
|使用デバイス：`DEVICE`|GPU|
|シード値|`random.seed(8)`<br>`np.random.seed(8)`<br>`torch.manual_seed(8)`|
|エポック数：`NUM_EPOCHES`|10|
|バッチサイズ：`BATCH_SIZE`|128|
|最適化アルゴリズム|Adam|
|学習率：`LEARNING_RATE`|0.0002|
|減衰率 beta1|0.5|
|生成器に入力するノイズ z の次数：`NUM_INPUT_NOIZE_Z`|62|
|特徴マップの枚数：`NUM_FEATURE_MAPS`|64|


#### ☆ 損失関数のグラフ（実行条件１）：`main_mnist.py`
![cGANforMNIST_Loss_epoches20_lr5e-05_batchsize128](https://user-images.githubusercontent.com/25688193/56843765-b4652000-68e0-11e9-9f50-c331b10a68fe.png)<br>
> 学習があるタイミングで突然不安定化しており、損失関数値がうまく収束していない？
> 学習率が大きすぎる問題？

#### ☆ 生成器から生成された自動生成画像（実行条件１）：`main_mnist.py`

- エポック数 : 1 / イテレーション回数：468<br>
![cGANforMNIST_Image_epoches0_iters468](https://user-images.githubusercontent.com/25688193/56843808-11f96c80-68e1-11e9-97f9-060955899be5.png)<br>
    - 数字０（クラスラベル０）<br>
        ![cGANforMNIST_Image0_epoches0_iters468](https://user-images.githubusercontent.com/25688193/56843876-00649480-68e2-11e9-9f03-342d6a72ecbd.png)<br>
    - 数字１（クラスラベル１）<br>
    - 数字２（クラスラベル２）<br>

- エポック数 : 2 / イテレーション回数：936<br>
![cGANforMNIST_Image_epoches1_iters936](https://user-images.githubusercontent.com/25688193/56843798-102fa900-68e1-11e9-9c17-6092c400e905.png)

- エポック数 : 3 / イテレーション回数：1404<br>
![cGANforMNIST_Image_epoches2_iters1404](https://user-images.githubusercontent.com/25688193/56843799-102fa900-68e1-11e9-8b4c-192d5be1857f.png)

- エポック数 : 4 / イテレーション回数：1872<br>
![cGANforMNIST_Image_epoches3_iters1872](https://user-images.githubusercontent.com/25688193/56843800-10c83f80-68e1-11e9-9618-857c8f34d002.png)

- エポック数 : 5 / イテレーション回数 : 2340<br>
![cGANforMNIST_Image_epoches4_iters2340](https://user-images.githubusercontent.com/25688193/56843801-10c83f80-68e1-11e9-804a-efd0c7693a77.png)

- エポック数 : 6 / イテレーション回数 : 2808<br>
![cGANforMNIST_Image_epoches5_iters2808](https://user-images.githubusercontent.com/25688193/56843802-10c83f80-68e1-11e9-9d2c-33c20affb8b0.png)

- エポック数 : 7 / イテレーション回数 : 3276<br>
![cGANforMNIST_Image_epoches6_iters3276](https://user-images.githubusercontent.com/25688193/56843803-10c83f80-68e1-11e9-9c86-01ed4282a70e.png)

- エポック数 : 8 / イテレーション回数 : 3744<br>
![cGANforMNIST_Image_epoches7_iters3744](https://user-images.githubusercontent.com/25688193/56843804-1160d600-68e1-11e9-8c0a-42a39391747a.png)

- エポック数 : 9 / イテレーション回数 : 4212<br>
![cGANforMNIST_Image_epoches8_iters4212](https://user-images.githubusercontent.com/25688193/56843805-1160d600-68e1-11e9-9d94-68bd2818707f.png)


- エポック数 : 10 / イテレーション回数 : 4680<br>
![cGANforMNIST_Image_epoches9_iters4680](https://user-images.githubusercontent.com/25688193/56843806-1160d600-68e1-11e9-84e0-259a9916d1fa.png)
    - 数字０（クラスラベル０）<br>
        ![cGANforMNIST_Image0_epoches9_iters4680](https://user-images.githubusercontent.com/25688193/56843878-05294880-68e2-11e9-9602-6762c7f50400.png)
    - 数字１（クラスラベル１）<br>
        ![cGANforMNIST_Image1_epoches9_iters4680](https://user-images.githubusercontent.com/25688193/56843899-39046e00-68e2-11e9-9a99-950819a4af9b.png)
    - 数字２（クラスラベル２）<br>
        ![cGANforMNIST_Image2_epoches9_iters4680](https://user-images.githubusercontent.com/25688193/56843904-44f03000-68e2-11e9-84e0-f73366613959.png)
    - 数字３（クラスラベル３）<br>
        ![cGANforMNIST_Image3_epoches9_iters4680](https://user-images.githubusercontent.com/25688193/56844118-4c193d00-68e6-11e9-9957-e934daedcc1c.png)
    - 数字４（クラスラベル４）<br>
        ![cGANforMNIST_Image4_epoches9_iters4680](https://user-images.githubusercontent.com/25688193/56844119-4d4a6a00-68e6-11e9-9aa7-ac070ab441e3.png)
    - 数字５（クラスラベル５）<br>
        ![cGANforMNIST_Image5_epoches9_iters4680](https://user-images.githubusercontent.com/25688193/56844121-50ddf100-68e6-11e9-88e6-5c0357dbd1b8.png)
    - 数字６（クラスラベル６）<br>
    - 数字７（クラスラベル７）<br>
    - 数字８（クラスラベル８）<br>
    - 数字９（クラスラベル９）<br>

---

### ◎ コードの実行結果：`main.py`

|パラメータ名|値（実行条件１）|値（実行条件２）|
|---|---|---|
|学習用データセット：`DATASET`|"MNIST"|"CIFAR-10"|
|使用デバイス：`DEVICE`|GPU|←|
|シード値|`random.seed(8)`<br>`np.random.seed(8)`<br>`torch.manual_seed(8)`|←|
|エポック数：`NUM_EPOCHES`|50|←|
|バッチサイズ：`BATCH_SIZE`|128|←|
|最適化アルゴリズム|Adam|←|
|学習率：`LEARNING_RATE`|0.00005|←|
|減衰率 beta1|0.5|←|
|生成器に入力するノイズ z の次数：`NUM_INPUT_NOIZE_Z`|100|100|
|入力画像のサイズ：`IMAGE_SIZE`|64|64|
|入力画像のチャンネル数：`NUM_CHANNELS`|1|3|
|特徴マップの枚数：`NUM_FEATURE_MAPS`|64|64|
|クラスラベルの個数：`NUM_CLASSES`|10|x|

#### ☆ 損失関数のグラフ（実行条件１）：`main.py`


#### ☆ 生成器から生成された自動生成画像（実行条件１）：`main.py`

- エポック数 : 1 / イテレーション回数：468<br>
<br>

- エポック数 : 2 / イテレーション回数：936<br>
<br>

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
