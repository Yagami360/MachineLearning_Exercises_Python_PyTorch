# Pix2Pix_PyTorch【実装中...】
pix2pix の PyTorch での実装。<br>

## ■ 項目 [Contents]
1. [動作環境](#動作環境)
1. [使用法](#使用法)
1. [コード説明＆実行結果](#コード説明＆実行結果)
1. [背景理論](https://github.com/Yagami360/My_NoteBook/blob/master/%E6%83%85%E5%A0%B1%E5%B7%A5%E5%AD%A6/%E6%83%85%E5%A0%B1%E5%B7%A5%E5%AD%A6_%E6%A9%9F%E6%A2%B0%E5%AD%A6%E7%BF%92_%E7%94%9F%E6%88%90%E3%83%A2%E3%83%87%E3%83%AB.md#pix2pix)


## ■ 動作環境

- Windows 10
- Geforce GTX1050 / VRAM:2GB
- Python : 3.6
- Anaconda : 5.0.1
- PyTorch : 1.0.1

## ■ 使用法

- データのダウンロード

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
DATASET_PATH = "./maps_custom"    # 学習用データセットへのパス
NUM_SAVE_STEP = 1             # 自動生成画像の保存間隔（エポック単位）

NUM_EPOCHES = 10               # エポック数（学習回数）
LEARNING_RATE = 0.0002        # 学習率
IMAGE_SIZE = 256              # 入力画像のサイズ（pixel単位）
NUM_CHANNELS = 3              # 入力画像のチャンネル数
NUM_FEATURE_MAPS = 64         # 特徴マップの枚数
BATCH_SIZE = 32               # ミニバッチサイズ

```


<a id="コード説明＆実行結果"></a>

## ■ コード説明＆実行結果

### ◎ コードの実行結果：`main.py`

|パラメータ名|値（実行条件１）|値（実行条件２）|値（実行条件３）|
|---|---|---|---|
|使用デバイス：`DEVICE`|GPU|←|←|
|シード値|`random.seed(8)`<br>`np.random.seed(8)`<br>`torch.manual_seed(8)`|←|
|エポック数：`NUM_EPOCHES`|10|←|
|バッチサイズ：`BATCH_SIZE`|1|←|
|最適化アルゴリズム|Adam|←|
|学習率：`LEARNING_RATE`|0.0002|←|
|減衰率 beta1|0.5|←|
|入力画像のサイズ：`IMAGE_SIZE`|256||
|入力画像のチャンネル数：`NUM_CHANNELS`|3|
|特徴マップの枚数：`NUM_FEATURE_MAPS`|64|64|

#### ☆ 損失関数のグラフ（実行条件１）

#### ☆ 生成画像（実行条件１）


## ■ デバッグ情報
