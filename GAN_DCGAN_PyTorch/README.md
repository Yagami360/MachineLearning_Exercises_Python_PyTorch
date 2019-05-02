# GAN_DCGAN_PyTorch
DCGAN の PyTorch での実装。

- 参考コード
    - [PyTorch/Tutorials >  DCGAN Tutorial](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html)
    - [PyTorch (12) Generative Adversarial Networks (MNIST) - 人工知能に関する断創録](http://aidiary.hatenablog.com/entry/20180304/1520172429)

## ■ 項目 [Contents]
1. [動作環境](#動作環境)
1. [使用法](#使用法)
1. [コードの実行結果](#コードの実行結果)
1. [背景理論](https://github.com/Yagami360/My_NoteBook/blob/master/%E6%83%85%E5%A0%B1%E5%B7%A5%E5%AD%A6/%E6%83%85%E5%A0%B1%E5%B7%A5%E5%AD%A6_%E6%A9%9F%E6%A2%B0%E5%AD%A6%E7%BF%92_%E7%94%9F%E6%88%90%E3%83%A2%E3%83%87%E3%83%AB.md#DCGAN)

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
RESULT_PATH = "./result_forMNIST"   # 結果を保存するディレクトリ
NUM_SAVE_STEP = 100           # 自動生成画像の保存間隔（イテレーション単位）

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
DATASET = "MNIST"            # データセットの種類（"MNIST" or "CIFAR-10"）
#DATASET = "CIFAR-10"          # データセットの種類（"MNIST" or "CIFAR-10"）
DATASET_PATH = "./dataset"    # 学習用データセットへのパス
RESULT_PATH = "./result_" + DATASET      # 結果を保存するディレクトリ
NUM_SAVE_STEP = 100             # 自動生成画像の保存間隔（イテレーション単位）

NUM_EPOCHES = 50              # エポック数（学習回数）
LEARNING_RATE = 0.00005       # 学習率
IMAGE_SIZE = 64               # 入力画像のサイズ（pixel単位）
NUM_CHANNELS = 1              # 入力画像のチャンネル数
NUM_FEATURE_MAPS = 64         # 特徴マップの枚数
BATCH_SIZE = 128              # ミニバッチサイズ
NUM_INPUT_NOIZE_Z = 100       # 生成器に入力するノイズ z の次数
```


<a id="コードの実行結果"></a>

## ■ コードの実行結果：`main_mnist.py`

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

### ◎ 損失関数のグラフ（実行条件１）：`main_mnist.py`

<!--
![DCGANforMNIST_Loss_epoches10_lr0 0002_batchsize128](https://user-images.githubusercontent.com/25688193/56814818-eb084f80-687a-11e9-967f-062388b8d90a.png)<br>
-->

### ◎ 生成器から生成された自動生成画像（実行条件１）：`main_mnist.py`

- エポック数 : 1 / イテレーション回数：100<br>
![DCGANforMNIST_Image_epoches0_iters100](https://user-images.githubusercontent.com/25688193/57061214-0d74ef80-6cf7-11e9-8022-3f3f4b5b3d6b.png)<br>

- エポック数 : 1 / イテレーション回数：200<br>
![DCGANforMNIST_Image_epoches0_iters200](https://user-images.githubusercontent.com/25688193/57061213-0cdc5900-6cf7-11e9-8046-5a6f25fb112f.png)<br>

- エポック数 : 1 / イテレーション回数：500<br>
![DCGANforMNIST_Image_epoches1_iters500](https://user-images.githubusercontent.com/25688193/57061220-0fd74980-6cf7-11e9-8200-78dc4b036c16.png)<br>

- エポック数 : 3 / イテレーション回数：1000<br>
![DCGANforMNIST_Image_epoches2_iters1000](https://user-images.githubusercontent.com/25688193/57061224-136ad080-6cf7-11e9-86d5-830c55e5d3b5.png)

- エポック数 : 4 / イテレーション回数：1500<br>
![DCGANforMNIST_Image_epoches3_iters1500](https://user-images.githubusercontent.com/25688193/57061316-588f0280-6cf7-11e9-8dba-fef271a1896e.png)

- エポック数 : 5 / イテレーション回数：2000<br>
![DCGANforMNIST_Image_epoches4_iters2000](https://user-images.githubusercontent.com/25688193/57061317-5af15c80-6cf7-11e9-9297-4c02218002ed.png)

- エポック数 : 6 / イテレーション回数：2500<br>
![DCGANforMNIST_Image_epoches5_iters2500](https://user-images.githubusercontent.com/25688193/57061395-a146bb80-6cf7-11e9-97df-6f035c3dc2bc.png)

- エポック数 : 7 / イテレーション回数：3000<br>
![DCGANforMNIST_Image_epoches6_iters3000](https://user-images.githubusercontent.com/25688193/57061474-ef5bbf00-6cf7-11e9-85d0-8ecaf34e8a3c.png)

- エポック数 : 8 / イテレーション回数：3500<br>
![DCGANforMNIST_Image_epoches7_iters3500](https://user-images.githubusercontent.com/25688193/57061477-f08cec00-6cf7-11e9-8418-1f31757539ea.png)

- エポック数 : 9 / イテレーション回数：4000<br>

- エポック数 : 10 / イテレーション回数：4500<br>


## ■ コードの実行結果：`main.py`

|パラメータ名|値（実行条件１）|値（実行条件２）|
|---|---|---|
|学習用データセット：`DATASET`|"MNIST"|"CIFAR-10"|
|使用デバイス：`DEVICE`|GPU|←|
|シード値|`random.seed(8)`<br>`np.random.seed(8)`<br>`torch.manual_seed(8)`|←|
|エポック数：`NUM_EPOCHES`|10|←|
|バッチサイズ：`BATCH_SIZE`|128|←|
|最適化アルゴリズム|Adam|←|
|学習率：`LEARNING_RATE`|0.00005|←|
|減衰率 beta1|0.5|←|
|生成器に入力するノイズ z の次数：`NUM_INPUT_NOIZE_Z`|100|100|
|入力画像のサイズ：`IMAGE_SIZE`|64|64|
|入力画像のチャンネル数：`NUM_CHANNELS`|1|3|
|特徴マップの枚数：`NUM_FEATURE_MAPS`|64|64|

### ◎ 損失関数のグラフ（実行条件１）
![DCGAN_Loss_epoches10_lr5e-05_batchsize128](https://user-images.githubusercontent.com/25688193/57061669-b53eed00-6cf8-11e9-8ece-f9d9c9e8563e.png)<br>

### ◎ 生成器から生成された自動生成画像（実行条件１）

![DCGAN_Image_epoches9_iters4680](https://user-images.githubusercontent.com/25688193/57061695-c7b92680-6cf8-11e9-967c-7c09fa3b05bf.gif)<br>

<!--
- エポック数 : 1 / イテレーション回数：468<br>
![DCGAN_Image_epoches0_iters468](https://user-images.githubusercontent.com/25688193/56815729-02e0d300-687d-11e9-8dea-a2bdce4ae4b9.png)<br>

- エポック数 : 2 / イテレーション回数：936<br>
![DCGAN_Image_epoches1_iters936](https://user-images.githubusercontent.com/25688193/56815628-cdd48080-687c-11e9-973c-bc0e6d188034.png)<br>

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
-->