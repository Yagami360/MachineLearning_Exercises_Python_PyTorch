# UNet_PyTorch
UNet の PyTorch での実装<br>
UNet によるセマンティックセグメンテーションを利用して、衛星画像から地図を生成する。<br>

- 参考コード
    - [GitHub/GunhoChoi/Kind-PyTorch-Tutorial12_Semantic_Segmentation/](https://github.com/GunhoChoi/Kind-PyTorch-Tutorial/tree/master/12_Semantic_Segmentation)


## ■ 項目 [Contents]
1. [動作環境](#動作環境)
1. [使用法](#使用法)
1. [コード説明＆実行結果](#コード説明＆実行結果)
1. [背景理論](https://github.com/Yagami360/My_NoteBook/blob/master/%E6%83%85%E5%A0%B1%E5%B7%A5%E5%AD%A6/%E6%83%85%E5%A0%B1%E5%B7%A5%E5%AD%A6_%E6%A9%9F%E6%A2%B0%E5%AD%A6%E7%BF%92_%E7%94%9F%E6%88%90%E3%83%A2%E3%83%87%E3%83%AB.md#UNet)

## ■ 動作環境

- Windows 10
- Geforce GTX1050 / VRAM:2GB
- Python : 3.6
- Anaconda : 5.0.1
- PyTorch : 1.0.1

## ■ 使用法

- データのダウンロード
    - GPU で駆動させるときは、消費 VRAM を抑えるために、ダウンロードした学習用データの一部を削除して学習しています。

```python
wget https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/maps.tar.gz
tar -xzvf maps.tar.gz
```

- 使用法
```
$ python main.py
```

- 設定可能な定数

```python
[main.py]
#DEVICE = "CPU"               # 使用デバイス ("CPU" or "GPU")
DEVICE = "GPU"                # 使用デバイス ("CPU" or "GPU")
DATASET_PATH = "./maps"       # 学習用データセットへのパス
NUM_SAVE_STEP = 1             # 自動生成画像の保存間隔（エポック単位）

NUM_EPOCHES = 10              # エポック数（学習回数）
LEARNING_RATE = 0.0002        # 学習率 (Default:0.0002)
BATCH_SIZE = 1                # ミニバッチサイズ
IMAGE_SIZE = 256              # 入力画像のサイズ（pixel単位）
NUM_FEATURE_MAPS = 64         # 特徴マップの枚数
```


<a id="コード説明＆実行結果"></a>

## ■ コード説明＆実行結果

### ◎ コードの実行結果：`main.py`

|パラメータ名|値（実行条件１）|値（実行条件２）|
|---|---|---|
|使用デバイス：`DEVICE`|GPU|←|
|シード値|`random.seed(8)`<br>`np.random.seed(8)`<br>`torch.manual_seed(8)`|←|
|エポック数：`NUM_EPOCHES`|10|50|
|バッチサイズ：`BATCH_SIZE`|1|←|
|特徴マップ数：`NUM_FEATURE_MAPS`|64|
|最適化アルゴリズム|Adam|←|
|学習率：`LEARNING_RATE`|0.0002|←|
|減衰率 beta1|0.5|←|

#### ☆ 損失関数のグラフ（実行条件１）
![UNet_Loss_epoches10_lr0 0002_batchsize1](https://user-images.githubusercontent.com/25688193/57000545-2d20f080-6bef-11e9-84c1-b2687067b04c.png)<br>

#### ☆ 生成画像（実行条件１）

- Epoch = 1 ; iteration = 50<br>
![UNet_Image_epoches0_iters50](https://user-images.githubusercontent.com/25688193/56968890-f0c5a400-6b9e-11e9-978a-dcfcbc115a06.png)<br>

- Epoch = 1 ; iteration = 100<br>
![UNet_Image_epoches0_iters100](https://user-images.githubusercontent.com/25688193/56968891-f15e3a80-6b9e-11e9-809b-a2589e15966b.png)<br>

- Epoch = 1 ; iteration = 500<br>
![UNet_Image_epoches0_iters500](https://user-images.githubusercontent.com/25688193/56969995-31262180-6ba1-11e9-9a06-aa80e018a3af.png)<br>

- Epoch = 1 ; iteration = 1096<br>
![UNet_Image_epoches0_iters1096](https://user-images.githubusercontent.com/25688193/56969999-34211200-6ba1-11e9-84cf-497b07965b46.png)<br>

- Epoch = 2 ; iteration = 2192<br>
![UNet_Image_epoches1_iters2192](https://user-images.githubusercontent.com/25688193/56969943-1b186100-6ba1-11e9-985c-fe138901becf.png)<br>

- Epoch = 3 ; iteration = 3288<br>
![UNet_Image_epoches2_iters3288](https://user-images.githubusercontent.com/25688193/56970130-734f6300-6ba1-11e9-9119-56cafc358ca3.png)<br>

- Epoch = 4 ; iteration = 4384<br>
![UNet_Image_epoches3_iters4384](https://user-images.githubusercontent.com/25688193/57000615-72452280-6bef-11e9-9636-42c69e788727.png)<br>

- Epoch = 5 ; iteration = 5480<br>
![UNet_Image_epoches4_iters5480](https://user-images.githubusercontent.com/25688193/57000602-6bb6ab00-6bef-11e9-9ec2-6c0ddec4f72e.png)<br>

- Epoch = 6 ; iteration = 6576<br>
![UNet_Image_epoches5_iters6576](https://user-images.githubusercontent.com/25688193/57000597-68232400-6bef-11e9-97a2-bd2250056b78.png)<br>

- Epoch = 7 ; iteration = 7672<br>
![UNet_Image_epoches6_iters7672](https://user-images.githubusercontent.com/25688193/57000594-648f9d00-6bef-11e9-9b7f-f9539d90ba5a.png)<br>

- Epoch = 8 ; iteration = 8768<br>
![UNet_Image_epoches7_iters8768](https://user-images.githubusercontent.com/25688193/57000589-60637f80-6bef-11e9-9bc6-bd74adb64691.png)<br>

- Epoch = 9 ; iteration = 9864<br>
![UNet_Image_epoches8_iters9864](https://user-images.githubusercontent.com/25688193/57000572-52adfa00-6bef-11e9-96f0-f47c9a1f448c.png)<br>

- Epoch = 10 ; iteration = 10960<br>
![UNet_Image_epoches9_iters10960](https://user-images.githubusercontent.com/25688193/57000557-42961a80-6bef-11e9-9941-00f80719191f.png)<br>

## ■ デバッグ情報
