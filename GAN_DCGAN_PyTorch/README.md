# 【実装中...】GAN_DCGAN_PyTorch
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
DEVICE = "GPU"                # 使用デバイス ("CPU" or "GPU")
DATASET_PATH = "./dataset"    # 学習用データセットへのパス
NUM_EPOCHES = 25              # エポック数（学習回数）
LEARNING_RATE = 0.0002        # 学習率
BATCH_SIZE = 32               # ミニバッチサイズ (Default:32)
```


<a id="コード説明＆実行結果"></a>

## ■ コード説明＆実行結果

### ◎ コードの実行結果

|パラメータ名|値（実行条件１）|値（実行条件２）|
|---|---|---|
|学習用データセット|MNIST|←|
|使用デバイス：`DEVICE`|GPU|←|
|エポック数：`NUM_EPOCHES`|25|←|
|バッチサイズ：`BATCH_SIZE`|32|←|
|最適化アルゴリズム|Adam|←|
|学習率：`LEARNING_RATE`|0.0002|←|
|xxx|xxx|xxx|