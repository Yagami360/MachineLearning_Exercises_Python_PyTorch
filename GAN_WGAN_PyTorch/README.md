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
DATASET_PATH = "./dataset"    # 学習用データセットへのパス
NUM_SAVE_STEP = 1             # 自動生成画像の保存間隔（エポック単位）

NUM_EPOCHES = 25              # エポック数（学習回数）
LEARNING_RATE = 0.0002        # 学習率
BATCH_SIZE = 128              # ミニバッチサイズ
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
|学習用データセット|MNIST|←|
|使用デバイス：`DEVICE`|GPU|←|
|シード値|`random.seed(8)`<br>`np.random.seed(8)`<br>`torch.manual_seed(8)`|←|
|エポック数：`NUM_EPOCHES`|10|←|
|バッチサイズ：`BATCH_SIZE`|128|←|
|最適化アルゴリズム|RMSProp|←|
|学習率：`LEARNING_RATE`|0.00005|←|
|生成器に入力するノイズ z の次数：`NUM_INPUT_NOIZE_Z`|62|←|
|クリティックの更新回数：`NUM_CRITIC`|5|
|重みクリッピングの下限値：`WEIGHT_CLAMP_LOWER`|-0.01|
|重みクリッピングの上限値：`WEIGHT_CLAMP_UPPER`|0.01|


#### ☆ 損失関数のグラフ（実行条件１）
<br>

#### ☆ 生成器から生成された自動生成画像（実行条件１）

- エポック数 : 1 / イテレーション回数：xxx<br>

- エポック数 : 2 / イテレーション回数：xxx<br>


- エポック数 : 10 / イテレーション回数 : xxxx<br>
<br>


## ■ デバッグ情報

```python
CGAN_G(
  (main): Sequential(
    (initial:100-512:convt): ConvTranspose2d(100, 512, kernel_size=(4, 4), stride=(1, 1), bias=False)
    (initial:512:batchnorm): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (initial:512:relu): ReLU(inplace)
    (pyramid:512-256:convt): ConvTranspose2d(512, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    (pyramid:256:batchnorm): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (pyramid:256:relu): ReLU(inplace)
    (pyramid:256-128:convt): ConvTranspose2d(256, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    (pyramid:128:batchnorm): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (pyramid:128:relu): ReLU(inplace)
    (pyramid:128-64:convt): ConvTranspose2d(128, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    (pyramid:64:batchnorm): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (pyramid:64:relu): ReLU(inplace)
    (final:64-3:convt): ConvTranspose2d(64, 3, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    (final:3:tanh): Tanh()
  )
)
DCGAN_D(
  (main): Sequential(
    (initial:3-64:conv): Conv2d(3, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    (initial:64:relu): LeakyReLU(negative_slope=0.2, inplace)
    (pyramid:64-128:conv): Conv2d(64, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    (pyramid:128:batchnorm): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (pyramid:128:relu): LeakyReLU(negative_slope=0.2, inplace)
    (pyramid:128-256:conv): Conv2d(128, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    (pyramid:256:batchnorm): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (pyramid:256:relu): LeakyReLU(negative_slope=0.2, inplace)
    (pyramid:256-512:conv): Conv2d(256, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    (pyramid:512:batchnorm): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (pyramid:512:relu): LeakyReLU(negative_slope=0.2, inplace)
    (final:512-1:conv): Conv2d(512, 1, kernel_size=(4, 4), stride=(1, 1), bias=False)
  )
)
```