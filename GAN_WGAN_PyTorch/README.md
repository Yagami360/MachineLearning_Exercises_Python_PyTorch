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
DATASET = "MNIST"             # データセットの種類（"MNIST" or "CIFAR-10"）
DATASET_PATH = "./dataset"    # 学習用データセットへのパス
NUM_SAVE_STEP = 1             # 自動生成画像の保存間隔（エポック単位）

NUM_EPOCHES = 25              # エポック数（学習回数）
LEARNING_RATE = 0.00005       # 学習率
BATCH_SIZE = 128              # ミニバッチサイズ
IMAGE_SIZE = 64               # 入力画像のサイズ（pixel単位）
NUM_CHANNELS = 3              # 入力画像のチャンネル数
NUM_FEATURE_MAPS = 64         # 特徴マップの枚数
NUM_INPUT_NOIZE_Z = 62        # 生成器に入力するノイズ z の次数
NUM_CRITIC = 5                # クリティックの更新回数
WEIGHT_CLAMP_LOWER = - 0.01   # 重みクリッピングの下限値
WEIGHT_CLAMP_UPPER = 0.01     # 重みクリッピングの上限値
```


<a id="コード説明＆実行結果"></a>

## ■ コード説明＆実行結果

### ◎ コードの実行結果

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


#### ☆ 損失関数のグラフ（実行条件１）
<br>

#### ☆ 生成器から生成された自動生成画像（実行条件１）

- エポック数 : 1 / イテレーション回数：937<br>
![WGAN_Image_epoches0_iters937](https://user-images.githubusercontent.com/25688193/56650043-3971ee80-66c1-11e9-9788-e21a045fd1f6.png)<br>

- エポック数 : 2 / イテレーション回数：1874<br>
![WGAN_Image_epoches1_iters1874](https://user-images.githubusercontent.com/25688193/56650044-3971ee80-66c1-11e9-9d79-9bf412420c55.png)<br>

- エポック数 : 3 / イテレーション回数 : 2811<br>
![WGAN_Image_epoches2_iters2811](https://user-images.githubusercontent.com/25688193/56650046-3a0a8500-66c1-11e9-92b2-664689014d73.png)<br>

- エポック数 : 4 / イテレーション回数 : 3749<br>
![WGAN_Image_epoches3_iters3748](https://user-images.githubusercontent.com/25688193/56650704-8d310780-66c2-11e9-8dc0-5b2fe87fb867.png)<br>

- エポック数 : 10 / イテレーション回数 : xxx<br>
<br>


## ■ デバッグ情報
