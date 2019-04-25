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
![WGAN_Loss_epoches10_lr5e-05_batchsize64](https://user-images.githubusercontent.com/25688193/56708937-9a4d0580-6759-11e9-860f-b29dabffc242.png)<br>
> DCGAN よりは安定しているが、乱高下があり、論文中のグラフと異なる？

#### ☆ 生成器から生成された自動生成画像（実行条件１）

- エポック数 : 1 / イテレーション回数：937<br>
![WGAN_Image_epoches0_iters937](https://user-images.githubusercontent.com/25688193/56708938-9a4d0580-6759-11e9-84e0-4d7b516e376c.png)<br>

- エポック数 : 2 / イテレーション回数：1874<br>
![WGAN_Image_epoches1_iters1874](https://user-images.githubusercontent.com/25688193/56708939-9ae59c00-6759-11e9-9268-f301394230c3.png)<br>

- エポック数 : 3 / イテレーション回数 : 2811<br>
![WGAN_Image_epoches2_iters2811](https://user-images.githubusercontent.com/25688193/56708940-9ae59c00-6759-11e9-8d06-6c023ca62058.png)<br>

- エポック数 : 4 / イテレーション回数 : 3748<br>
![WGAN_Image_epoches3_iters3748](https://user-images.githubusercontent.com/25688193/56708941-9b7e3280-6759-11e9-8c45-7212b12b502b.png)<br>

- エポック数 : 5 / イテレーション回数 : 4685<br>
![WGAN_Image_epoches4_iters4685](https://user-images.githubusercontent.com/25688193/56708943-9b7e3280-6759-11e9-944e-9de7c21cb518.png)

- エポック数 : 6 / イテレーション回数 : 5622<br>
![WGAN_Image_epoches5_iters5622](https://user-images.githubusercontent.com/25688193/56708944-9b7e3280-6759-11e9-9e4f-ca79a3252946.png)<br>

- エポック数 : 7 / イテレーション回数 : 6559<br>
![WGAN_Image_epoches6_iters6559](https://user-images.githubusercontent.com/25688193/56708945-9c16c900-6759-11e9-9fd1-97a23363a9ba.png)<br>

- エポック数 : 8 / イテレーション回数 : 7496<br>
![WGAN_Image_epoches7_iters7496](https://user-images.githubusercontent.com/25688193/56708946-9c16c900-6759-11e9-96e9-a3d39d171de5.png)<br>

- エポック数 : 9 / イテレーション回数 : 8433<br>
![WGAN_Image_epoches8_iters8433](https://user-images.githubusercontent.com/25688193/56708947-9caf5f80-6759-11e9-8247-00c64730b52c.png)<br>

- エポック数 : 10 / イテレーション回数 : 9370<br>
![WGAN_Image_epoches9_iters9370](https://user-images.githubusercontent.com/25688193/56708936-99b46f00-6759-11e9-8cbe-4a63420d714a.png)<br>


## ■ デバッグ情報
