# Pix2Pix_PyTorch
pix2pix の PyTorch での実装。<br>
pix2pix によるセマンティックセグメンテーションを利用して、衛星画像から地図を生成する。<br>

- 参考コード
    - [PyTorch-GAN/implementations/pix2pix/](https://github.com/eriklindernoren/PyTorch-GAN/tree/master/implementations/pix2pix)


## ■ 項目 [Contents]
1. [動作環境](#動作環境)
1. [使用法](#使用法)
1. [コードの実行結果](#コードの実行結果)
1. [背景理論](https://github.com/Yagami360/My_NoteBook/blob/master/%E6%83%85%E5%A0%B1%E5%B7%A5%E5%AD%A6/%E6%83%85%E5%A0%B1%E5%B7%A5%E5%AD%A6_%E6%A9%9F%E6%A2%B0%E5%AD%A6%E7%BF%92_%E7%94%9F%E6%88%90%E3%83%A2%E3%83%87%E3%83%AB.md#pix2pix)


## ■ 動作環境

- Windows 10
- Geforce GTX1050 / VRAM:2GB
- Python : 3.6
- Anaconda : 5.0.1
- PyTorch : 1.0.1

## ■ 使用法

- 使用データ（航空写真と地図画像）<br>
https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/maps.tar.gz<br>
よりダウンロード後、解凍。

- 使用法
```
$ python main.py
```

- 設定可能な定数

```python
[main.py]
#DEVICE = "CPU"               # 使用デバイス ("CPU" or "GPU")
DEVICE = "GPU"                # 使用デバイス ("CPU" or "GPU")
DATASET_PATH = "./maps_custom"    # 学習用データセットへのパス
NUM_SAVE_STEP = 100           # 自動生成画像の保存間隔（イテレーション単位）

NUM_EPOCHES = 10               # エポック数（学習回数）
LEARNING_RATE = 0.0002        # 学習率
IMAGE_SIZE = 256              # 入力画像のサイズ（pixel単位）
NUM_CHANNELS = 3              # 入力画像のチャンネル数
NUM_FEATURE_MAPS = 64         # 特徴マップの枚数
BATCH_SIZE = 32               # ミニバッチサイズ

```


<a id="コード説明＆実行結果"></a>

## ■ コードの実行結果：`main.py`

|パラメータ名|値（実行条件１）|値（実行条件２）|
|---|---|---|
|使用デバイス：`DEVICE`|GPU|←|←|
|シード値|`random.seed(8)`<br>`np.random.seed(8)`<br>`torch.manual_seed(8)`|←|
|エポック数：`NUM_EPOCHES`|10|←|
|バッチサイズ：`BATCH_SIZE`|1|←|
|最適化アルゴリズム|Adam|←|
|学習率：`LEARNING_RATE`|0.0002||
|減衰率 beta1|0.5|←|
|入力画像のサイズ：`IMAGE_SIZE`|256||
|入力画像のチャンネル数：`NUM_CHANNELS`|3|←|
|特徴マップの枚数：`NUM_FEATURE_MAPS`|64||

※ GPU で駆動させるときは、消費 VRAM を抑えるために、ダウンロードした学習用データの内、一部のみ（1.jpg ~ 200.jpg の 200枚）を使用して学習している。

### ◎ 損失関数のグラフ（実行条件１）
![Pix2Pix_Loss_epoches10_lr0 0002](https://user-images.githubusercontent.com/25688193/57012858-726f0d80-6c43-11e9-8fc0-23c76a5cdff8.png)<br>

> ０付近の値に安定的に収束しており、うまく学習していることが見てとれる。

### ◎ 生成画像（実行条件１）

- Epoch 1 : iterations = 50<br>
![UNet_Image_epoches0_iters50](https://user-images.githubusercontent.com/25688193/57012140-0a6af800-6c40-11e9-91d1-a5df91f16094.png)<br>

- Epoch 1 : iterations = 100<br>
![UNet_Image_epoches0_iters100](https://user-images.githubusercontent.com/25688193/57012141-0a6af800-6c40-11e9-9189-c58581d357b9.png)<br>

- Epoch 1 : iterations = 200<br>
![Pix2Pix_Image_epoches0_iters200](https://user-images.githubusercontent.com/25688193/57012686-b7467480-6c42-11e9-9fc4-a74f46de0644.png)

- Epoch 2 : iterations = 400<br>
![Pix2Pix_Image_epoches1_iters400](https://user-images.githubusercontent.com/25688193/57012687-b7467480-6c42-11e9-9add-58501730fde6.png)

- Epoch 3 : iterations = 600<br>
![Pix2Pix_Image_epoches2_iters600](https://user-images.githubusercontent.com/25688193/57012688-b7467480-6c42-11e9-9738-5690e664954b.png)

- Epoch 4 : iterations = 800<br>
![Pix2Pix_Image_epoches3_iters800](https://user-images.githubusercontent.com/25688193/57012690-b7df0b00-6c42-11e9-8f9e-5ebe8c8ea117.png)

- Epoch 5 : iterations = 1000<br>
![Pix2Pix_Image_epoches4_iters1000](https://user-images.githubusercontent.com/25688193/57012691-b7df0b00-6c42-11e9-943c-a21eb69c0c54.png)

- Epoch 6 : iterations = 1200<br>
![Pix2Pix_Image_epoches5_iters1200](https://user-images.githubusercontent.com/25688193/57012692-b7df0b00-6c42-11e9-8181-20b87a97b7d2.png)

- Epoch 7 : iterations = 1400<br>
![Pix2Pix_Image_epoches6_iters1400](https://user-images.githubusercontent.com/25688193/57012694-b7df0b00-6c42-11e9-9b48-8a78b1ad5923.png)

- Epoch 8 : iterations = 1600<br>
![Pix2Pix_Image_epoches7_iters1600](https://user-images.githubusercontent.com/25688193/57012695-b877a180-6c42-11e9-8d0c-36a23dcd0093.png)

- Epoch 9 : iterations = 1800<br>
![Pix2Pix_Image_epoches8_iters1800](https://user-images.githubusercontent.com/25688193/57012855-6f741d00-6c43-11e9-94d4-5294e80928ee.png)<br>

- Epoch 10 : iterations = 2000<br>
![Pix2Pix_Image_epoches9_iters2000](https://user-images.githubusercontent.com/25688193/57012854-6edb8680-6c43-11e9-9396-ce0d048fe424.png)<br>


## ■ デバッグ情報

```python
_dicriminator : Discriminator(
  (_layer): Sequential(
    (0): Conv2d(6, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
    (1): LeakyReLU(negative_slope=0.2, inplace)
    (2): Conv2d(64, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
    (3): InstanceNorm2d(128, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
    (4): LeakyReLU(negative_slope=0.2, inplace)
    (5): Conv2d(128, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
    (6): InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
    (7): LeakyReLU(negative_slope=0.2, inplace)
    (8): Conv2d(256, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
    (9): InstanceNorm2d(512, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
    (10): LeakyReLU(negative_slope=0.2, inplace)
    (11): ZeroPad2d(padding=(1, 0, 1, 0), value=0.0)
    (12): Conv2d(512, 1, kernel_size=(4, 4), stride=(1, 1), padding=(1, 1), bias=False)
  )
)
```