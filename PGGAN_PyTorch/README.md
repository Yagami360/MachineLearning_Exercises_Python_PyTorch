# PGGAN_PyTorch
ProgressiveGAN（PGGAN）の PyTorch での実装。<br>

- 参考コード
    - [GitHub/github-pengge/PyTorch-progressive_growing_of_gans](https://github.com/github-pengge/PyTorch-progressive_growing_of_gans)
    - [GitHub/jeromerony/Progressive_Growing_of_GANs-PyTorch](https://github.com/jeromerony/Progressive_Growing_of_GANs-PyTorch)


## ■ 項目 [Contents]
1. [動作環境](#動作環境)
1. [使用法](#使用法)
1. [コードの実行結果](#コードの実行結果)
1. [背景理論](https://github.com/Yagami360/My_NoteBook/blob/master/%E6%83%85%E5%A0%B1%E5%B7%A5%E5%AD%A6/%E6%83%85%E5%A0%B1%E5%B7%A5%E5%AD%A6_%E6%A9%9F%E6%A2%B0%E5%AD%A6%E7%BF%92_%E7%94%9F%E6%88%90%E3%83%A2%E3%83%87%E3%83%AB.md#ProgressiveGAN%EF%BC%88PGGAN%EF%BC%89)


## ■ 動作環境

- Windows 10
- Geforce GTX1050 / VRAM:2GB
- Python : 3.6
- Anaconda : 5.0.1
- PyTorch : 1.0.1

## ■ 使用法

<!--
- 使用データ

  - CelebA<br>
  https://drive.google.com/drive/folders/0B7EVK8r0v71pWEZsZE9oNnFzTm8<br>
  よりダウンロード後、解凍。

  - CelebHD
  https://drive.google.com/drive/folders/0B4qLcYyJmiz0TXY1NG02bzZVRGs
  よりダウンロード後、解凍。

- 事前処理
  - `python2 h5tool.py create_celeba_hq [file_name_to_save] [/path/to/celeba_dataset/] [/path/to/celeba_hq_deltas]`
-->

- 使用法
```
$ python main.py
```

- 設定可能なコマンドライン引数

|引数名|意味|値 (Default値)|
|---|---|---|
|`--device`|実行デバイス|`GPU` (Default) or `CPU`|
|`--run_mode`|動作モード|`train` (Default) or `add_train` or `test`|
|`--dataset`|データセット|`MNIST` (Default) or `CIFAR-10`|
|`--dataset_path`|データセットの保存先|`./dataset` (Default)|
|`--n_input_noize_z`|入力ノイズ z の次元数（＝潜在変数の次元数）|`128`(Default)|
|`--init_image_size`|最初の Training Progresses での生成画像の解像度|`4`(Default)|
|`--final_image_size`|最終的な Training Progresses での生成画像の解像度|`32`(Default)|
|`--n_epoches`|エポック数|`10` (Default)|
|`--batch_size`|バッチサイズ|`32` (Default)|
|`--learning_rate`|学習率|`0.001` (Default)|
|`--result_path`|学習結果のフォルダ|`./result` (Default)|
|`--xxx`|xxx|`xxx` (Default)|


<a id="コード説明＆実行結果"></a>

## ■ コードの実行結果：`main.py`

|パラメータ名|値（実行条件１）|値（実行条件２）|
|---|---|---|
|使用デバイス：<br>`--device`|GPU|←|
|データセット：<br>`--dataset`|MNIST|CIFAR-10|
|動作モード：<br>`--run_mode`|`train`|←|
|エポック数：<br>`--n_epoches`|10|10|
|入力ノイズ z の次元数：<br>`--n_input_noize_z`|128|256|
|最初の生成画像の解像度：<br>`--init_image_size`|4|4|
|最終的なの生成画像の解像度：<br>`--final_image_size`|32|64|
|バッチサイズ：<br>`--batch_size`|16|16|
|最適化アルゴリズム|Adam|←|
|学習率：<br>`--learning_rate`|0.001|
|減衰率 beta1|0.5|←|
|減衰率 beta2|0.999|←|
|簡略化された Minibatch discrimation|有り|←|
|Equalized learning rate|有り|←|
|Pixel norm|有り|←|
|シード値|`random.seed(8)`<br>`np.random.seed(8)`<br>`torch.manual_seed(8)`|←|


### ◎ 損失関数のグラフ（実行条件１）

![PGGAN_Loss_epoches10_lr0 001_batchsize16](https://user-images.githubusercontent.com/25688193/59812205-1de53600-9348-11e9-96bb-a6b37b92a201.png)

- iterations = 0 ~ 3750 : Traing Progress 1 (4 × 4 pixel) <br>
- iterations = 3751 ~ 11250 : Traing Progress 2 (8 × 8 pixel) <br>
- iterations = 11251 ~ 18750 : Traing Progress 3 (16 × 16 pixel) <br>
- iterations = 18751 ~ 37500 : Traing Progress 4 (32 × 32 pixel) <br>


### ◎ 生成画像（実行条件１）

- [Traing Progress 1] : 4 × 4 pixel 縦（8 枚）×横（８枚）<br>
	- Epoch 0 : iterations = 3750<br>
		![PGGAN_Image_iters3750](https://user-images.githubusercontent.com/25688193/59812366-a82d9a00-9348-11e9-9140-aab74e331c8e.png)<br>

- [Traing Progress 2] : 8 × 8 pixel 縦（8 枚）×横（８枚）
	- Epoch 1 : iterations = 11250<br>
		![PGGAN_Image_iters11250](https://user-images.githubusercontent.com/25688193/59812425-e925ae80-9348-11e9-98a7-ddc8b2e8d93b.png)<br>

- [Traing Progress 3] : 16 × 16 pixel 縦（8 枚）×横（８枚）
	- Epoch 2 : iterations = 12000<br>
		![PGGAN_Image_iters12000](https://user-images.githubusercontent.com/25688193/59812577-77019980-9349-11e9-942c-b03b5474b6d3.png)<br>

	- Epoch 3 : iterations = 18750<br>
		![PGGAN_Image_iters18750](https://user-images.githubusercontent.com/25688193/59812564-681ae700-9349-11e9-9d1c-8d2be6820163.png)<br>

- [Traing Progress 4] : 32 × 32 pixel 縦（8 枚）×横（８枚）
	- Epoch 4 : iterations = 19000<br>
		![PGGAN_Image_iters19000](https://user-images.githubusercontent.com/25688193/59812644-b0d2a000-9349-11e9-96ba-7f3cb092ae6b.png)<br>

	- Epoch 5 : iterations = 23000<br>
		![PGGAN_Image_iters23000](https://user-images.githubusercontent.com/25688193/59812748-06a74800-934a-11e9-9bb6-1bed332fab52.png)<br>

	- Epoch 9 : iterations = 37500<br>
		![PGGAN_Image_iters37500](https://user-images.githubusercontent.com/25688193/59812701-d2cc2280-9349-11e9-8705-4c26fff84712.png)<br>

### ◎ 損失関数のグラフ（実行条件２）

- iterations = 0 ~ 3750 : Traing Progress 1 (4 × 4 pixel) <br>
- iterations = 3751 ~ 11250 : Traing Progress 2 (8 × 8 pixel) <br>
- iterations = 11251 ~ 18750 : Traing Progress 3 (16 × 16 pixel) <br>
- iterations = 18751 ~ 37500 : Traing Progress 4 (32 × 32 pixel) <br>


### ◎ 生成画像（実行条件２）

- [Traing Progress 1] : 4 × 4 pixel 縦（8 枚）×横（８枚）<br>
	- Epoch 0 : iterations = 3750<br>


- [Traing Progress 2] : 8 × 8 pixel 縦（8 枚）×横（８枚）<br>
	- Epoch 1 : iterations = xxx<br>

- [Traing Progress 3] : 16 × 16 pixel 縦（8 枚）×横（８枚）<br>
	- Epoch 2 : iterations = xxx<br>

- [Traing Progress 4] : 32 × 32 pixel 縦（8 枚）×横（８枚）<br>
	- Epoch 3 : iterations = xxx<br>

- [Traing Progress 5] : 64 × 64 pixel 縦（8 枚）×横（８枚）<br>
	- Epoch 4 : iterations = xxx<br>
	- Epoch 5 : iterations = xxx<br>
	- Epoch 9 : iterations = xxx<br>


## ■ デバッグ情報


`$ python h5tool.py create_celeba_hq /Users/sakai/ML_dataset/CelebA-HQ /Users/sakai/ML_dataset/CelebA/celebA /Users/sakai/ML_dataset/CelebA/celebA-HQ`

```python
Generator(
  (output_layer): GSelectLayer(
    (pre): PixelNormLayer(eps = 1e-08)
    (chain): ModuleList(
      (0): Sequential(
        (0): ReshapeLayer()
        (1): Conv2d(512, 512, kernel_size=(4, 4), stride=(1, 1), padding=(3, 3), bias=False)
        (2): WScaleLayer(incoming = Conv2d)
        (3): LeakyReLU(negative_slope=0.2)
        (4): PixelNormLayer(eps = 1e-08)
        (5): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (6): WScaleLayer(incoming = Conv2d)
        (7): LeakyReLU(negative_slope=0.2)
        (8): PixelNormLayer(eps = 1e-08)
      )
      (1): Sequential(
        (0): Upsample(scale_factor=2, mode=nearest)
        (1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (2): WScaleLayer(incoming = Conv2d)
        (3): LeakyReLU(negative_slope=0.2)
        (4): PixelNormLayer(eps = 1e-08)
        (5): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (6): WScaleLayer(incoming = Conv2d)
        (7): LeakyReLU(negative_slope=0.2)
        (8): PixelNormLayer(eps = 1e-08)
      )
      (2): Sequential(
        (0): Upsample(scale_factor=2, mode=nearest)
        (1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (2): WScaleLayer(incoming = Conv2d)
        (3): LeakyReLU(negative_slope=0.2)
        (4): PixelNormLayer(eps = 1e-08)
        (5): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (6): WScaleLayer(incoming = Conv2d)
        (7): LeakyReLU(negative_slope=0.2)
        (8): PixelNormLayer(eps = 1e-08)
      )
      (3): Sequential(
        (0): Upsample(scale_factor=2, mode=nearest)
        (1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (2): WScaleLayer(incoming = Conv2d)
        (3): LeakyReLU(negative_slope=0.2)
        (4): PixelNormLayer(eps = 1e-08)
        (5): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (6): WScaleLayer(incoming = Conv2d)
        (7): LeakyReLU(negative_slope=0.2)
        (8): PixelNormLayer(eps = 1e-08)
      )
      (4): Sequential(
        (0): Upsample(scale_factor=2, mode=nearest)
        (1): Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (2): WScaleLayer(incoming = Conv2d)
        (3): LeakyReLU(negative_slope=0.2)
        (4): PixelNormLayer(eps = 1e-08)
        (5): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (6): WScaleLayer(incoming = Conv2d)
        (7): LeakyReLU(negative_slope=0.2)
        (8): PixelNormLayer(eps = 1e-08)
      )
      (5): Sequential(
        (0): Upsample(scale_factor=2, mode=nearest)
        (1): Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (2): WScaleLayer(incoming = Conv2d)
        (3): LeakyReLU(negative_slope=0.2)
        (4): PixelNormLayer(eps = 1e-08)
        (5): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (6): WScaleLayer(incoming = Conv2d)
        (7): LeakyReLU(negative_slope=0.2)
        (8): PixelNormLayer(eps = 1e-08)
      )
      (6): Sequential(
        (0): Upsample(scale_factor=2, mode=nearest)
        (1): Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (2): WScaleLayer(incoming = Conv2d)
        (3): LeakyReLU(negative_slope=0.2)
        (4): PixelNormLayer(eps = 1e-08)
        (5): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (6): WScaleLayer(incoming = Conv2d)
        (7): LeakyReLU(negative_slope=0.2)
        (8): PixelNormLayer(eps = 1e-08)
      )
    )
    (post): ModuleList(
      (0): Sequential(
        (0): Conv2d(512, 3, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): WScaleLayer(incoming = Conv2d)
      )
      (1): Sequential(
        (0): Conv2d(512, 3, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): WScaleLayer(incoming = Conv2d)
      )
      (2): Sequential(
        (0): Conv2d(512, 3, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): WScaleLayer(incoming = Conv2d)
      )
      (3): Sequential(
        (0): Conv2d(512, 3, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): WScaleLayer(incoming = Conv2d)
      )
      (4): Sequential(
        (0): Conv2d(256, 3, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): WScaleLayer(incoming = Conv2d)
      )
      (5): Sequential(
        (0): Conv2d(128, 3, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): WScaleLayer(incoming = Conv2d)
      )
      (6): Sequential(
        (0): Conv2d(64, 3, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): WScaleLayer(incoming = Conv2d)
      )
    )
  )
)
```

```python
Discriminator(
  (output_layer): DSelectLayer(
    (chain): ModuleList(
      (0): Sequential(
        (0): GDropLayer(mode = prop, strength = 0.0, axes = [0, 1], normalize = False)
        (1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (2): WScaleLayer(incoming = Conv2d)
        (3): LeakyReLU(negative_slope=0.2)
        (4): GDropLayer(mode = prop, strength = 0.0, axes = [0, 1], normalize = False)
        (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (6): WScaleLayer(incoming = Conv2d)
        (7): LeakyReLU(negative_slope=0.2)
        (8): AvgPool2d(kernel_size=2, stride=2, padding=0)
      )
      (1): Sequential(
        (0): GDropLayer(mode = prop, strength = 0.0, axes = [0, 1], normalize = False)
        (1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (2): WScaleLayer(incoming = Conv2d)
        (3): LeakyReLU(negative_slope=0.2)
        (4): GDropLayer(mode = prop, strength = 0.0, axes = [0, 1], normalize = False)
        (5): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (6): WScaleLayer(incoming = Conv2d)
        (7): LeakyReLU(negative_slope=0.2)
        (8): AvgPool2d(kernel_size=2, stride=2, padding=0)
      )
      (2): Sequential(
        (0): GDropLayer(mode = prop, strength = 0.0, axes = [0, 1], normalize = False)
        (1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (2): WScaleLayer(incoming = Conv2d)
        (3): LeakyReLU(negative_slope=0.2)
        (4): GDropLayer(mode = prop, strength = 0.0, axes = [0, 1], normalize = False)
        (5): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (6): WScaleLayer(incoming = Conv2d)
        (7): LeakyReLU(negative_slope=0.2)
        (8): AvgPool2d(kernel_size=2, stride=2, padding=0)
      )
      (3): Sequential(
        (0): GDropLayer(mode = prop, strength = 0.0, axes = [0, 1], normalize = False)
        (1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (2): WScaleLayer(incoming = Conv2d)
        (3): LeakyReLU(negative_slope=0.2)
        (4): GDropLayer(mode = prop, strength = 0.0, axes = [0, 1], normalize = False)
        (5): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (6): WScaleLayer(incoming = Conv2d)
        (7): LeakyReLU(negative_slope=0.2)
        (8): AvgPool2d(kernel_size=2, stride=2, padding=0)
      )
      (4): Sequential(
        (0): GDropLayer(mode = prop, strength = 0.0, axes = [0, 1], normalize = False)
        (1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (2): WScaleLayer(incoming = Conv2d)
        (3): LeakyReLU(negative_slope=0.2)
        (4): GDropLayer(mode = prop, strength = 0.0, axes = [0, 1], normalize = False)
        (5): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (6): WScaleLayer(incoming = Conv2d)
        (7): LeakyReLU(negative_slope=0.2)
        (8): AvgPool2d(kernel_size=2, stride=2, padding=0)
      )
      (5): Sequential(
        (0): GDropLayer(mode = prop, strength = 0.0, axes = [0, 1], normalize = False)
        (1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (2): WScaleLayer(incoming = Conv2d)
        (3): LeakyReLU(negative_slope=0.2)
        (4): GDropLayer(mode = prop, strength = 0.0, axes = [0, 1], normalize = False)
        (5): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (6): WScaleLayer(incoming = Conv2d)
        (7): LeakyReLU(negative_slope=0.2)
        (8): AvgPool2d(kernel_size=2, stride=2, padding=0)
      )
      (6): Sequential(
        (0): MinibatchStatConcatLayer(averaging = all)
        (1): GDropLayer(mode = prop, strength = 0.0, axes = [0, 1], normalize = False)
        (2): Conv2d(513, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (3): WScaleLayer(incoming = Conv2d)
        (4): LeakyReLU(negative_slope=0.2)
        (5): GDropLayer(mode = prop, strength = 0.0, axes = [0, 1], normalize = False)
        (6): Conv2d(512, 512, kernel_size=(4, 4), stride=(1, 1), bias=False)
        (7): WScaleLayer(incoming = Conv2d)
        (8): LeakyReLU(negative_slope=0.2)
        (9): Conv2d(512, 1, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (10): WScaleLayer(incoming = Conv2d)
        (11): Sigmoid()
      )
    )
    (inputs): ModuleList(
      (0): Sequential(
        (0): Conv2d(3, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): WScaleLayer(incoming = Conv2d)
        (2): LeakyReLU(negative_slope=0.2)
      )
      (1): Sequential(
        (0): Conv2d(3, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): WScaleLayer(incoming = Conv2d)
        (2): LeakyReLU(negative_slope=0.2)
      )
      (2): Sequential(
        (0): Conv2d(3, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): WScaleLayer(incoming = Conv2d)
        (2): LeakyReLU(negative_slope=0.2)
      )
      (3): Sequential(
        (0): Conv2d(3, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): WScaleLayer(incoming = Conv2d)
        (2): LeakyReLU(negative_slope=0.2)
      )
      (4): Sequential(
        (0): Conv2d(3, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): WScaleLayer(incoming = Conv2d)
        (2): LeakyReLU(negative_slope=0.2)
      )
      (5): Sequential(
        (0): Conv2d(3, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): WScaleLayer(incoming = Conv2d)
        (2): LeakyReLU(negative_slope=0.2)
      )
      (6): Sequential(
        (0): Conv2d(3, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): WScaleLayer(incoming = Conv2d)
        (2): LeakyReLU(negative_slope=0.2)
      )
    )
  )
)
```

```python
Generator(
  (toRGBs): ModuleList(
    (0): Sequential(
      (conv): Conv2d(128, 1, kernel_size=(1, 1), stride=(1, 1))
    )
    (1): Sequential(
      (conv): Conv2d(128, 1, kernel_size=(1, 1), stride=(1, 1))
    )
    (2): Sequential(
      (conv): Conv2d(128, 1, kernel_size=(1, 1), stride=(1, 1))
    )
    (3): Sequential(
      (conv): Conv2d(128, 1, kernel_size=(1, 1), stride=(1, 1))
    )
  )
  (blocks): ModuleList(
    (0): Sequential(
      (conv0): Sequential(
        (conv): Conv2d(128, 128, kernel_size=(4, 4), stride=(1, 1), padding=(3, 3))
        (activ): LeakyReLU(negative_slope=0.2)
      )
      (conv1): Sequential(
        (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (activ): LeakyReLU(negative_slope=0.2)
      )
    )
    (1): Sequential(
      (conv0): Sequential(
        (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (activ): LeakyReLU(negative_slope=0.2)
      )
      (conv1): Sequential(
        (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (activ): LeakyReLU(negative_slope=0.2)
      )
    )
    (2): Sequential(
      (conv0): Sequential(
        (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (activ): LeakyReLU(negative_slope=0.2)
      )
      (conv1): Sequential(
        (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (activ): LeakyReLU(negative_slope=0.2)
      )
    )
    (3): Sequential(
      (conv0): Sequential(
        (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (activ): LeakyReLU(negative_slope=0.2)
      )
      (conv1): Sequential(
        (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (activ): LeakyReLU(negative_slope=0.2)
      )
    )
  )
)
Discriminator(
  (fromRGBs): ModuleList(
    (0): Sequential(
      (conv): Conv2d(1, 128, kernel_size=(1, 1), stride=(1, 1))
      (activ): LeakyReLU(negative_slope=0.2)
    )
    (1): Sequential(
      (conv): Conv2d(1, 128, kernel_size=(1, 1), stride=(1, 1))
      (activ): LeakyReLU(negative_slope=0.2)
    )
    (2): Sequential(
      (conv): Conv2d(1, 128, kernel_size=(1, 1), stride=(1, 1))
      (activ): LeakyReLU(negative_slope=0.2)
    )
    (3): Sequential(
      (conv): Conv2d(1, 128, kernel_size=(1, 1), stride=(1, 1))
      (activ): LeakyReLU(negative_slope=0.2)
    )
  )
  (blocks): ModuleList(
    (0): Sequential(
      (conv_std): Sequential(
        (conv): Conv2d(129, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (activ): LeakyReLU(negative_slope=0.2)
      )
      (conv_pool): Sequential(
        (conv): Conv2d(128, 128, kernel_size=(4, 4), stride=(1, 1))
        (activ): LeakyReLU(negative_slope=0.2)
      )
      (conv_class): Sequential(
        (conv): Conv2d(128, 1, kernel_size=(1, 1), stride=(1, 1))
      )
    )
    (1): Sequential(
      (conv0): Sequential(
        (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (activ): LeakyReLU(negative_slope=0.2)
      )
      (conv1): Sequential(
        (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (activ): LeakyReLU(negative_slope=0.2)
      )
    )
    (2): Sequential(
      (conv0): Sequential(
        (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (activ): LeakyReLU(negative_slope=0.2)
      )
      (conv1): Sequential(
        (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (activ): LeakyReLU(negative_slope=0.2)
      )
    )
    (3): Sequential(
      (conv0): Sequential(
        (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (activ): LeakyReLU(negative_slope=0.2)
      )
      (conv1): Sequential(
        (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (activ): LeakyReLU(negative_slope=0.2)
      )
    )
  )

```