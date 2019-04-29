# GAN_WGAN_PyTorch
Wasserstein GAN（WGAN）の PyTorch での実装。

- 参考コード
    - [martinarjovsky/WassersteinGAN（元論文の実装）](https://github.com/martinarjovsky/WassersteinGAN)
    - [tjwei/GANotebooks/wgan-torch.ipynb](https://github.com/tjwei/GANotebooks/blob/master/wgan-torch.ipynb)


## ■ 項目 [Contents]
1. [動作環境](#動作環境)
1. [使用法](#使用法)
1. [コード説明＆実行結果](#コード説明＆実行結果)
1. [背景理論](https://github.com/Yagami360/My_NoteBook/blob/master/%E6%83%85%E5%A0%B1%E5%B7%A5%E5%AD%A6/%E6%83%85%E5%A0%B1%E5%B7%A5%E5%AD%A6_%E6%A9%9F%E6%A2%B0%E5%AD%A6%E7%BF%92_%E7%94%9F%E6%88%90%E3%83%A2%E3%83%87%E3%83%AB.md#WGAN)

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
DATASET = "MNIST"            # データセットの種類（"MNIST" or "CIFAR-10"）
#DATASET = "CIFAR-10"          # データセットの種類（"MNIST" or "CIFAR-10"）
DATASET_PATH = "./dataset"    # 学習用データセットへのパス
NUM_SAVE_STEP = 1             # 自動生成画像の保存間隔（エポック単位）

NUM_EPOCHES = 10               # エポック数（学習回数）
LEARNING_RATE = 0.00005       # 学習率 (Default:0.00005)
BATCH_SIZE = 64               # ミニバッチサイズ
IMAGE_SIZE = 64               # 入力画像のサイズ（pixel単位）
NUM_CHANNELS = 1              # 入力画像のチャンネル数
NUM_FEATURE_MAPS = 64         # 特徴マップの枚数
NUM_INPUT_NOIZE_Z = 100       # 生成器に入力するノイズ z の次数
NUM_CRITIC = 5                # クリティックの更新回数
WEIGHT_CLAMP_LOWER = - 0.01   # 重みクリッピングの下限値
WEIGHT_CLAMP_UPPER = 0.01     # 重みクリッピングの上限値
```

<!--
```python
[main_mnist.py]
#DEVICE = "CPU"               # 使用デバイス ("CPU" or "GPU")
DEVICE = "GPU"                # 使用デバイス ("CPU" or "GPU")
DATASET_PATH = "./dataset"    # 学習用データセットへのパス
NUM_SAVE_STEP = 1             # 自動生成画像の保存間隔（エポック単位）

NUM_EPOCHES = 10              # エポック数（学習回数）
LEARNING_RATE = 0.0002        # 学習率 (Default:0.00005)
BATCH_SIZE = 64               # ミニバッチサイズ
NUM_FEATURE_MAPS = 64         # 特徴マップの枚数
NUM_INPUT_NOIZE_Z = 62        # 生成器に入力するノイズ z の次数
NUM_CRITIC = 5                # クリティックの更新回数
WEIGHT_CLAMP_LOWER = - 0.01   # 重みクリッピングの下限値
WEIGHT_CLAMP_UPPER = 0.01     # 重みクリッピングの上限値
```
-->

<a id="コード説明＆実行結果"></a>

## ■ コード説明＆実行結果

### ◎ コードの実行結果：`main.py`

|パラメータ名|値（実行条件１）|値（実行条件２）|
|---|---|---|
|学習用データセット：`DATASET`|"MNIST"|"CIFAR-10"|
|使用デバイス：`DEVICE`|GPU|←|
|シード値|`random.seed(8)`<br>`np.random.seed(8)`<br>`torch.manual_seed(8)`|←|
|エポック数：`NUM_EPOCHES`|10|50|
|バッチサイズ：`BATCH_SIZE`|64|64|
|生成器に入力するノイズ z の次数：`NUM_INPUT_NOIZE_Z`|100|100|
|入力画像のサイズ：`IMAGE_SIZE`|64|64|
|入力画像のチャンネル数：`NUM_CHANNELS`|1|3|
|特徴マップの枚数：`NUM_FEATURE_MAPS`|64|64|
|最適化アルゴリズム|Adam|←|
|学習率：`LEARNING_RATE`|0.00005|←|
|クリティックの更新回数：`NUM_CRITIC`|5|←|
|重みクリッピングの下限値：`WEIGHT_CLAMP_LOWER`|-0.01|←|
|重みクリッピングの上限値：`WEIGHT_CLAMP_UPPER`|0.01|←|


#### ☆ 損失関数のグラフ（実行条件１）：`main.py`
![WGAN_Loss_epoches10_lr5e-05_batchsize64](https://user-images.githubusercontent.com/25688193/56844723-05c8db80-68f0-11e9-8fd3-9f4692c4e27c.png)<br>

<!--
> DCGAN よりは安定しているが、乱高下があり、論文中のグラフと異なる？
-->

#### ☆ 生成器から生成された自動生成画像（実行条件１）：`main.py`

- エポック数 : 1 / イテレーション回数：937<br>
![WGAN_Image_epoches0_iters937](https://user-images.githubusercontent.com/25688193/56844476-9e109180-68eb-11e9-91d9-469c63d82825.png)<br>

- エポック数 : 2 / イテレーション回数：1874<br>
![WGAN_Image_epoches1_iters1874](https://user-images.githubusercontent.com/25688193/56844477-9e109180-68eb-11e9-8503-01f70a512847.png)<br>

- エポック数 : 3 / イテレーション回数 : 2811<br>
![WGAN_Image_epoches2_iters2811](https://user-images.githubusercontent.com/25688193/56844478-9ea92800-68eb-11e9-97c6-3e3242a10202.png)<br>

- エポック数 : 4 / イテレーション回数 : 3748<br>
![WGAN_Image_epoches3_iters3748](https://user-images.githubusercontent.com/25688193/56844473-9d77fb00-68eb-11e9-83f4-ea6681ca6ce1.png)<br>

- エポック数 : 5 / イテレーション回数 : 4685<br>
![WGAN_Image_epoches4_iters4685](https://user-images.githubusercontent.com/25688193/56844474-9d77fb00-68eb-11e9-91d0-f8c19699e22e.png)<br>

- エポック数 : 6 / イテレーション回数 : 5622<br>
![WGAN_Image_epoches5_iters5622](https://user-images.githubusercontent.com/25688193/56844475-9e109180-68eb-11e9-970a-dbc9a4f1e29f.png)<br>

- エポック数 : 7 / イテレーション回数 : 6559<br>
![WGAN_Image_epoches6_iters6559](https://user-images.githubusercontent.com/25688193/56844499-fa73b100-68eb-11e9-82ee-bd3512fd13b2.png)<br>

- エポック数 : 8 / イテレーション回数 : 7496<br>
![WGAN_Image_epoches7_iters7496](https://user-images.githubusercontent.com/25688193/56844598-c8634e80-68ed-11e9-93e9-2401d2909c6f.png)<br>

- エポック数 : 9 / イテレーション回数 : 8433<br>
![WGAN_Image_epoches8_iters8433](https://user-images.githubusercontent.com/25688193/56844599-c8634e80-68ed-11e9-9bd8-0718ee102c0d.png)<br>

- エポック数 : 10 / イテレーション回数 : 9370<br>
![WGAN_Image_epoches9_iters9370](https://user-images.githubusercontent.com/25688193/56844720-f3e73880-68ef-11e9-9c0f-fe86df550736.png)<br>

---

<!--
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
|クリティックの更新回数：`NUM_CRITIC`|5|←|
|重みクリッピングの下限値：`WEIGHT_CLAMP_LOWER`|-0.01|←|
|重みクリッピングの上限値：`WEIGHT_CLAMP_UPPER`|0.01|←|

#### ☆ 損失関数のグラフ（実行条件１）：`main_mnist.py`
![WGANforMNIST_Loss_epoches10_lr0 0002_batchsize64](https://user-images.githubusercontent.com/25688193/56844042-1031a800-68e5-11e9-833d-7307db54b21f.png)<br>

#### ☆ 生成器から生成された自動生成画像（実行条件１）：`main_mnist.py`

- エポック数 : 1 / イテレーション回数：937<br>
![WGANforMNIST_Image_epoches0_iters937](https://user-images.githubusercontent.com/25688193/56844069-5555da00-68e5-11e9-8290-055a686cbbed.png)<br>

- エポック数 : 2 / イテレーション回数：1874<br>
![WGANforMNIST_Image_epoches1_iters1874](https://user-images.githubusercontent.com/25688193/56844070-5555da00-68e5-11e9-9ffe-cc4bf0047515.png)<br>

- エポック数 : 3 / イテレーション回数 : 2811<br>
![WGANforMNIST_Image_epoches2_iters2811](https://user-images.githubusercontent.com/25688193/56844060-538c1680-68e5-11e9-8cb3-971d000ac756.png)<br>

- エポック数 : 4 / イテレーション回数 : 3748<br>
![WGANforMNIST_Image_epoches3_iters3748](https://user-images.githubusercontent.com/25688193/56844061-5424ad00-68e5-11e9-84f5-f2b6613e7c35.png)<br>

- エポック数 : 5 / イテレーション回数 : 4685<br>
![WGANforMNIST_Image_epoches4_iters4685](https://user-images.githubusercontent.com/25688193/56844062-5424ad00-68e5-11e9-81c5-5501fae3a420.png)<br>

- エポック数 : 6 / イテレーション回数 : 5622<br>
![WGANforMNIST_Image_epoches5_iters5622](https://user-images.githubusercontent.com/25688193/56844063-5424ad00-68e5-11e9-8c4d-2626bb333c19.png)<br>

- エポック数 : 7 / イテレーション回数 : 6559<br>
![WGANforMNIST_Image_epoches6_iters6559](https://user-images.githubusercontent.com/25688193/56844064-54bd4380-68e5-11e9-835d-30efaaaeba2d.png)<br>

- エポック数 : 8 / イテレーション回数 : 7496<br>
![WGANforMNIST_Image_epoches7_iters7496](https://user-images.githubusercontent.com/25688193/56844065-54bd4380-68e5-11e9-80bd-704e936be65f.png)<br>

- エポック数 : 9 / イテレーション回数 : 8433<br>
![WGANforMNIST_Image_epoches8_iters8433](https://user-images.githubusercontent.com/25688193/56844066-54bd4380-68e5-11e9-8949-c0dc73bd2a7f.png)<br>

- エポック数 : 10 / イテレーション回数 : 9370<br>
![WGANforMNIST_Image_epoches9_iters9370](https://user-images.githubusercontent.com/25688193/56844067-5555da00-68e5-11e9-9bd9-d010db367729.png)<br>
-->


## ■ デバッグ情報
