# GAN_DCGAN_PyTorch
DCGAN の PyTorch での実装。

## ■ 項目 [Contents]
1. [動作環境](#動作環境)
1. [使用法](#使用法)
1. [コード説明＆実行結果](#コード説明＆実行結果)
1. 背景理論

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
|最適化アルゴリズム|Adam|←|
|学習率：`LEARNING_RATE`|0.0002|←|
|減衰率 beta1|0.5|←|
|生成器に入力するノイズ z の次数：`NUM_INPUT_NOIZE_Z`|62|←|

#### ☆ 損失関数のグラフ（実行条件１）
![DCGAN_Loss_epoches10_lr0 0002_batchsize128](https://user-images.githubusercontent.com/25688193/55666851-430cf100-588f-11e9-99f9-ad31f1dd0034.png)<br>

#### ☆ 生成器から生成された自動生成画像（実行条件１）

- エポック数 : 1 / イテレーション回数：468<br>
![DCGAN_Image_epoches0_iters468](https://user-images.githubusercontent.com/25688193/55666728-e52bd980-588d-11e9-862f-6747d242797d.png)<br>

- エポック数 : 2 / イテレーション回数：936<br>
![DCGAN_Image_epoches1_iters936](https://user-images.githubusercontent.com/25688193/55666736-f70d7c80-588d-11e9-8a8b-0671864750dd.png)<br>

- エポック数 : 3 / イテレーション回数：1404<br>
![DCGAN_Image_epoches2_iters1404](https://user-images.githubusercontent.com/25688193/55666758-36d46400-588e-11e9-9428-ae7e9225028f.png)<br>

- エポック数 : 4 / イテレーション回数：1872<br>
![DCGAN_Image_epoches3_iters1872](https://user-images.githubusercontent.com/25688193/55666765-4489e980-588e-11e9-9713-bc7117428e22.png)<br>

- エポック数 : 5 / イテレーション回数 : 2340<br>
![DCGAN_Image_epoches4_iters2340](https://user-images.githubusercontent.com/25688193/55666777-7a2ed280-588e-11e9-8460-533c4009f414.png)<br>

- エポック数 : 10 / イテレーション回数 : 4680<br>
![DCGAN_Image_epoches9_iters4680](https://user-images.githubusercontent.com/25688193/55666854-515b0d00-588f-11e9-8810-122407d71309.png)<br>
