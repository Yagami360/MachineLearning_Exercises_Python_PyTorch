# GAN_cGAN_PyTorch
Conditional GAN（cGAN）の PyTorch での実装。<br>

ネットワーク構成は、CNN を使用（DCGAN or LSGANベース）

## ■ 項目 [Contents]
1. [動作環境](#動作環境)
1. [使用法](#使用法)
1. [コード説明＆実行結果](#コード説明＆実行結果)
1. [背景理論](https://github.com/Yagami360/My_NoteBook/blob/master/%E6%83%85%E5%A0%B1%E5%B7%A5%E5%AD%A6/%E6%83%85%E5%A0%B1%E5%B7%A5%E5%AD%A6_%E6%A9%9F%E6%A2%B0%E5%AD%A6%E7%BF%92_%E7%94%9F%E6%88%90%E3%83%A2%E3%83%87%E3%83%AB.md#ConditionalGAN%EF%BC%88CGAN%EF%BC%89)

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

#GAN_BASELINE = "DCGAN"       # GAN のベースラインアルゴリズム（"DCGAN" or "LSGAN"）
GAN_BASELINE = "LSGAN"        # GAN のベースラインアルゴリズム（"DCGAN" or "LSGAN"）

NUM_EPOCHES = 50              # エポック数（学習回数）
LEARNING_RATE = 0.00005       # 学習率
BATCH_SIZE = 128              # ミニバッチサイズ
IMAGE_SIZE = 64               # 入力画像のサイズ（pixel単位）
NUM_CHANNELS = 1              # 入力画像のチャンネル数
NUM_FEATURE_MAPS = 64         # 特徴マップの枚数
NUM_INPUT_NOIZE_Z = 100       # 生成器に入力するノイズ z の次数
NUM_CLASSES = 10              # クラスラベル y の次元数
```


<a id="コード説明＆実行結果"></a>

## ■ コード説明＆実行結果

### ◎ コードの実行結果：`main.py`

|パラメータ名|値（実行条件１）|値（実行条件２）|値（実行条件３）|
|---|---|---|---|
|学習用データセット：`DATASET`|"MNIST"|←|"CIFAR-10"|
|使用デバイス：`DEVICE`|GPU|←|←|
|シード値|`random.seed(8)`<br>`np.random.seed(8)`<br>`torch.manual_seed(8)`|←|
|GAN のベースラインアルゴリズム（"DCGAN" or "LSGAN"）：`GAN_BASELINE`|"DCGAN"|"LSGAN"|
|エポック数：`NUM_EPOCHES`|10|←|
|バッチサイズ：`BATCH_SIZE`|128|←|
|最適化アルゴリズム|Adam|←|
|学習率：`LEARNING_RATE`|0.00005|←|
|減衰率 beta1|0.5|←|
|生成器に入力するノイズ z の次数：`NUM_INPUT_NOIZE_Z`|100|100|
|入力画像のサイズ：`IMAGE_SIZE`|64|64|
|入力画像のチャンネル数：`NUM_CHANNELS`|1|1|3|
|特徴マップの枚数：`NUM_FEATURE_MAPS`|64|64|
|クラスラベルの個数：`NUM_CLASSES`|10|10|x|

#### ☆ 損失関数のグラフ（実行条件１）：`main.py`

- 学習率：0.0002
![cGAN_Loss_epoches10_lr0 0002_batchsize128](https://user-images.githubusercontent.com/25688193/56874668-81ed2b80-6a76-11e9-925d-f6e02e8cda5d.png)<br>

- 学習率：0.00005
![cGAN_Loss_epoches10_lr5e-05_batchsize128](https://user-images.githubusercontent.com/25688193/56875412-bd3e2900-6a7b-11e9-894a-4ed09480bfcb.png)<br>


#### ☆ 生成器から生成された自動生成画像（実行条件１）：`main.py`

##### 学習率：0.00005

- エポック数 : 1 / イテレーション回数：468<br>
![cDCGAN_Image_epoches0_iters468](https://user-images.githubusercontent.com/25688193/56875710-8bc65d00-6a7d-11e9-847e-a91e5ec79b93.png)<br>
    - 数字０（クラスラベル０）<br>
        ![cDCGAN_Image0_epoches0_iters468](https://user-images.githubusercontent.com/25688193/56875821-1909b180-6a7e-11e9-82ce-62a636af6ea8.png)<br>
    - 数字１（クラスラベル１）<br>
        ![cDCGAN_Image1_epoches0_iters468](https://user-images.githubusercontent.com/25688193/56875825-19a24800-6a7e-11e9-920c-af5a1a63cd52.png)<br>
    - 数字２（クラスラベル２）<br>
        ![cDCGAN_Image2_epoches0_iters468](https://user-images.githubusercontent.com/25688193/56875828-1ad37500-6a7e-11e9-88ab-7a6d03fc6ab8.png)<br>
    - 数字３（クラスラベル３）<br>
        ![cDCGAN_Image3_epoches0_iters468](https://user-images.githubusercontent.com/25688193/56875833-1eff9280-6a7e-11e9-9098-1ec34941e708.png)<br>

- エポック数 : 2 / イテレーション回数：936<br>
![cDCGAN_Image_epoches1_iters936](https://user-images.githubusercontent.com/25688193/56875711-8c5ef380-6a7d-11e9-87b9-26e117374e8e.png)<br>
    - 数字０（クラスラベル０）<br>
        ![cDCGAN_Image0_epoches1_iters936](https://user-images.githubusercontent.com/25688193/56875823-1909b180-6a7e-11e9-8159-18200c88ead3.png)<br>
    - 数字１（クラスラベル１）<br>
        ![cDCGAN_Image1_epoches1_iters936](https://user-images.githubusercontent.com/25688193/56875826-1a3ade80-6a7e-11e9-983d-8c2fe003d91d.png)<br>
    - 数字２（クラスラベル２）<br>
        ![cDCGAN_Image2_epoches1_iters936](https://user-images.githubusercontent.com/25688193/56875831-1d35cf00-6a7e-11e9-90ba-fd04e1851d4f.png)<br>
    - 数字３（クラスラベル３）<br>
        ![cDCGAN_Image3_epoches1_iters936](https://user-images.githubusercontent.com/25688193/56875834-1f982900-6a7e-11e9-991b-6b6b5c2ba28d.png)<br>

- エポック数 : 3 / イテレーション回数：1404<br>
![cDCGAN_Image_epoches2_iters1404](https://user-images.githubusercontent.com/25688193/56875712-8c5ef380-6a7d-11e9-9968-3818b8962fca.png)<br>

- エポック数 : 4 / イテレーション回数：1872<br>
![cDCGAN_Image_epoches3_iters1872](https://user-images.githubusercontent.com/25688193/56875713-8c5ef380-6a7d-11e9-98fd-4fd0fc91e1ff.png)<br>

- エポック数 : 5 / イテレーション回数 : 2340<br>
![cDCGAN_Image_epoches4_iters2340](https://user-images.githubusercontent.com/25688193/56875714-8cf78a00-6a7d-11e9-83f3-865c738a9b0b.png)<br>

- エポック数 : 6 / イテレーション回数 : 2808<br>
![cDCGAN_Image_epoches5_iters2808](https://user-images.githubusercontent.com/25688193/56875715-8d902080-6a7d-11e9-8a4c-5d8146a60409.png)<br>

- エポック数 : 7 / イテレーション回数 : 3276<br>
![cDCGAN_Image_epoches6_iters3276](https://user-images.githubusercontent.com/25688193/56875716-8d902080-6a7d-11e9-96b0-ac452d8b406c.png)<br>

- エポック数 : 8 / イテレーション回数 : 3744<br>
![cDCGAN_Image_epoches7_iters3744](https://user-images.githubusercontent.com/25688193/56875718-8ec14d80-6a7d-11e9-88c7-b230e708e743.png)<br>

- エポック数 : 9 / イテレーション回数 : 4212<br>
![cDCGAN_Image_epoches8_iters4212](https://user-images.githubusercontent.com/25688193/56875719-8ec14d80-6a7d-11e9-9754-544240a097ed.png)<br>

- エポック数 : 10 / イテレーション回数 : 4680<br>
![cDCGAN_Image_epoches9_iters4680](https://user-images.githubusercontent.com/25688193/56875708-8bc65d00-6a7d-11e9-8483-f6b92f240bda.png)<br>
    - 数字０（クラスラベル０）<br>
        ![cDCGAN_Image0_epoches9_iters4680](https://user-images.githubusercontent.com/25688193/56875824-19a24800-6a7e-11e9-997e-eecaa8ad8250.png)<br>
    - 数字１（クラスラベル１）<br>
        ![cDCGAN_Image1_epoches9_iters4680](https://user-images.githubusercontent.com/25688193/56875827-1ad37500-6a7e-11e9-896d-bc0473937aa5.png)<br>
    - 数字２（クラスラベル２）<br>
        ![cDCGAN_Image2_epoches9_iters4680](https://user-images.githubusercontent.com/25688193/56875832-1e66fc00-6a7e-11e9-86e1-c31983a3c943.png)<br>
    - 数字３（クラスラベル３）<br>
        ![cDCGAN_Image3_epoches9_iters4680](https://user-images.githubusercontent.com/25688193/56875909-aea54100-6a7e-11e9-8b51-3ad5f4a97287.png)<br>
    - 数字４（クラスラベル４）<br>
        ![cDCGAN_Image4_epoches9_iters4680](https://user-images.githubusercontent.com/25688193/56875910-aea54100-6a7e-11e9-8e03-3baee665c7da.png)<br>
    - 数字５（クラスラベル５）<br>
        ![cDCGAN_Image5_epoches9_iters4680](https://user-images.githubusercontent.com/25688193/56875911-aea54100-6a7e-11e9-82b3-da478a62f157.png)<br>
    - 数字６（クラスラベル６）<br>
        ![cDCGAN_Image6_epoches9_iters4680](https://user-images.githubusercontent.com/25688193/56875912-af3dd780-6a7e-11e9-80f9-22392cd7ff87.png)<br>
    - 数字７（クラスラベル７）<br>
        ![cDCGAN_Image7_epoches9_iters4680](https://user-images.githubusercontent.com/25688193/56875915-af3dd780-6a7e-11e9-98c7-6c0ca8381c3a.png)<br>
    - 数字８（クラスラベル８）<br>
        ![cDCGAN_Image8_epoches9_iters4680](https://user-images.githubusercontent.com/25688193/56875916-afd66e00-6a7e-11e9-9532-9e0e69ee296e.png)<br>
    - 数字９（クラスラベル９）<br>
        ![cDCGAN_Image9_epoches9_iters4680](https://user-images.githubusercontent.com/25688193/56875918-afd66e00-6a7e-11e9-9004-bf187b732322.png)<br>


##### 学習率：0.0002

- エポック数 : 1 / イテレーション回数：468<br>
![cDCGAN_Image_epoches0_iters468](https://user-images.githubusercontent.com/25688193/56874420-b7911500-6a74-11e9-8559-133cddd19f3a.png)<br>
    - 数字０（クラスラベル０）<br>
        ![cDCGAN_Image0_epoches0_iters468](https://user-images.githubusercontent.com/25688193/56874485-16ef2500-6a75-11e9-9685-b989a5864e1c.png)<br>
    - 数字１（クラスラベル１）<br>
        ![cDCGAN_Image1_epoches0_iters468](https://user-images.githubusercontent.com/25688193/56874483-16ef2500-6a75-11e9-9444-5a11432f12e0.png)<br>
    - 数字２（クラスラベル２）<br>
        ![cDCGAN_Image2_epoches0_iters468](https://user-images.githubusercontent.com/25688193/56874486-18205200-6a75-11e9-8b2e-e31de2e002a4.png)<br>

- エポック数 : 2 / イテレーション回数：936<br>
![cDCGAN_Image_epoches1_iters936](https://user-images.githubusercontent.com/25688193/56874417-b6f87e80-6a74-11e9-9092-1e5ae71d1e6e.png)<br>
    - 数字０（クラスラベル０）<br>
        ![cDCGAN_Image0_epoches1_iters936](https://user-images.githubusercontent.com/25688193/56874521-56b60c80-6a75-11e9-8a64-e5a4bc0be6a0.png)<br>
    - 数字１（クラスラベル１）<br>
        ![cDCGAN_Image1_epoches1_iters936](https://user-images.githubusercontent.com/25688193/56874524-57e73980-6a75-11e9-96cb-39039f2e305e.png)<br>
    - 数字２（クラスラベル２）<br>
        ![cDCGAN_Image2_epoches1_iters936](https://user-images.githubusercontent.com/25688193/56874526-59186680-6a75-11e9-90a9-0488b8c04a74.png)<br>

- エポック数 : 3 / イテレーション回数：1404<br>
![cDCGAN_Image_epoches2_iters1404](https://user-images.githubusercontent.com/25688193/56874418-b7911500-6a74-11e9-8930-14d6b6a7d172.png)<br>

- エポック数 : 4 / イテレーション回数：1872<br>
![cDCGAN_Image_epoches3_iters1872](https://user-images.githubusercontent.com/25688193/56874450-ed35fe00-6a74-11e9-9df8-9b189738ee10.png)<br>

- エポック数 : 5 / イテレーション回数 : 2340<br>
![cDCGAN_Image_epoches4_iters2340](https://user-images.githubusercontent.com/25688193/56874555-98df4e00-6a75-11e9-8896-d9e96ce5d8b0.png)<br>

- エポック数 : 6 / イテレーション回数 : 2808<br>
![cDCGAN_Image_epoches5_iters2808](https://user-images.githubusercontent.com/25688193/56874556-98df4e00-6a75-11e9-95b9-a10bbe6325d0.png)<br>

- エポック数 : 7 / イテレーション回数 : 3276<br>
![cDCGAN_Image_epoches6_iters3276](https://user-images.githubusercontent.com/25688193/56874557-9977e480-6a75-11e9-8f74-c84813970a5d.png)<br>

- エポック数 : 8 / イテレーション回数 : 3744<br>
![cDCGAN_Image_epoches7_iters3744](https://user-images.githubusercontent.com/25688193/56874579-bdd3c100-6a75-11e9-9351-1cc92f1d3990.png)<br>

- エポック数 : 9 / イテレーション回数 : 4212<br>
![cDCGAN_Image_epoches8_iters4212](https://user-images.githubusercontent.com/25688193/56874600-e2c83400-6a75-11e9-8bf5-ce31383cd04d.png)<br>

- エポック数 : 10 / イテレーション回数 : 4680<br>
![cDCGAN_Image_epoches9_iters4680](https://user-images.githubusercontent.com/25688193/56874659-684be400-6a76-11e9-815e-7e415b1190fd.png)<br>
    - 数字０（クラスラベル０）<br>
        ![cDCGAN_Image0_epoches9_iters4680](https://user-images.githubusercontent.com/25688193/56874734-df817800-6a76-11e9-8ce2-97d53b6a1728.png)<br>
    - 数字１（クラスラベル１）<br>
        ![cDCGAN_Image1_epoches9_iters4680](https://user-images.githubusercontent.com/25688193/56874726-da242d80-6a76-11e9-9be4-d362d8e70e6a.png)<br>
    - 数字２（クラスラベル２）<br>
        ![cDCGAN_Image2_epoches9_iters4680](https://user-images.githubusercontent.com/25688193/56874721-d98b9700-6a76-11e9-9292-2ba540d343eb.png)<br>
    - 数字３（クラスラベル３）<br>
        ![cDCGAN_Image3_epoches9_iters4680](https://user-images.githubusercontent.com/25688193/56874722-d98b9700-6a76-11e9-9204-2e58bcd2a384.png)<br>
    - 数字４（クラスラベル４）<br>
        ![cDCGAN_Image4_epoches9_iters4680](https://user-images.githubusercontent.com/25688193/56874723-da242d80-6a76-11e9-9962-3f30d68face1.png)<br>
    - 数字５（クラスラベル５）<br>
        ![cDCGAN_Image5_epoches9_iters4680](https://user-images.githubusercontent.com/25688193/56874725-da242d80-6a76-11e9-9ce2-4856d8538a79.png)<br>
    - 数字６（クラスラベル６）<br>
        ![cDCGAN_Image6_epoches9_iters4680](https://user-images.githubusercontent.com/25688193/56874727-dabcc400-6a76-11e9-885b-e235a88bfdf9.png)<br>
    - 数字７（クラスラベル７）<br>
        ![cDCGAN_Image7_epoches9_iters4680](https://user-images.githubusercontent.com/25688193/56874728-dabcc400-6a76-11e9-9517-61aa3324c4d6.png)<br>
    - 数字８（クラスラベル８）<br>
        ![cDCGAN_Image8_epoches9_iters4680](https://user-images.githubusercontent.com/25688193/56874729-db555a80-6a76-11e9-8591-87c84b378d2e.png)<br>
    - 数字９（クラスラベル９）<br>
        ![cDCGAN_Image9_epoches9_iters4680](https://user-images.githubusercontent.com/25688193/56874730-dbedf100-6a76-11e9-9ffa-e4a82e04a527.png)<br>

## ■ デバッグ情報
