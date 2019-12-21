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

- Mac OS / ubuntu server
- Python : 3.6
- Anaconda : 5.0.1
- PyTorch : 1.1.0

## ■ 使用法

- 学習処理
  ```sh
  # （例１） train DCGAN for MNIST datset using GPU0, using networks only for mnist
  $ python train.py \
    --exper_name DCGAN_MNISTNet_train \
    --dataset mnist --image_size 64 \
    --networkG_type mnist --networkD_type mnist \
    --n_input_noize_z 62
  ```

  ```sh
  # （例２） train DCGAN for MNIST datset using GPU0, using networks for general purpose
  $ python train.py \
    --exper_name DCGAN_train \
    --dataset mnist --image_size 64 \
    --networkG_type vanilla --networkD_type PatchGAN
  ```

  ```sh
  # （例３） train DCGAN for cifar10 datset using GPU0
  $ python train.py \
    --exper_name DCGAN_train \
    --dataset cifar-10 --image_size 64 \
    --networkG_type vanilla --networkD_type PatchGAN
  ```

- 推論処理（実装中...）
  ```sh
  $ python test.py
  ```

- TensorBoard
  ```sh
  $ tensorboard --logdir ${TENSOR_BOARD_DIR} --port ${AVAILABLE_POOT}
  ```

  ```sh
  #（例）
  $ tensorboard --logdir tensorboard --port 6006
  ```

<a id="コードの実行結果"></a>

## ■ コードの実行結果

<!--

|パラメータ名|値（実行条件１）|値（実行条件２）|
|---|---|---|
|実験名：<br>`args.exper_name`|""|""|
|学習用データセット：`args.dataset`|"mnist"|"cifar-10"|
|使用デバイス：<br>`args.device`|"gpu"|←|
|シード値|`random.seed(8)`<br>`np.random.seed(8)`<br>`torch.manual_seed(8)`|←|
|エポック数：<br>`args.n_epoches`|10|50|
|バッチサイズ：<br>`args.batch_size`|64|64|
|生成器に入力するノイズ z の次数：<br>`args.n_input_noize_z`|100|100|
|入力画像のサイズ：<br>`args.image_size`|64|64|
|入力画像のチャンネル数：<br>`args.n_channels`|1|3|
|特徴マップの枚数：<br>`args.n_fmaps`|64|64|
|最適化アルゴリズム|Adam|←|
|学習率：<br>`args.lr`|0.00005|←|
|クリティックの更新回数：<br>`args.n_critic`|5|←|
|重みクリッピングの下限値：<br>`args.w_clamp_lower`|-0.01|←|
|重みクリッピングの上限値：<br>`args.w_clamp_upper`|0.01|←|

-->

### ◎ 損失関数のグラフ


### ◎ 生成器から生成された自動生成画像


## ■ デバッグ情報

```python
[Generator]
model_G :
 Generator(
  (layer): Sequential(
    (0): ConvTranspose2d(100, 512, kernel_size=(4, 4), stride=(1, 1), bias=False)
    (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU(inplace)
    (3): ConvTranspose2d(512, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    (4): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (5): ReLU(inplace)
    (6): ConvTranspose2d(256, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    (7): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (8): ReLU(inplace)
    (9): ConvTranspose2d(128, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    (10): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (11): ReLU(inplace)
    (12): ConvTranspose2d(64, 1, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    (13): Tanh()
  )
)
```

```python
[Generator for MNIST]
model_G :
 MNISTGenerator(
  (fc_layer): Sequential(
    (0): Linear(in_features=100, out_features=1024, bias=True)
    (1): BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU()
    (3): Linear(in_features=1024, out_features=6272, bias=True)
    (4): BatchNorm1d(6272, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (5): ReLU()
  )
  (deconv_layer): Sequential(
    (0): ConvTranspose2d(128, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
    (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU()
    (3): ConvTranspose2d(64, 1, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
    (4): Sigmoid()
  )
)
```

```python
model_D :
 PatchGANDiscriminator(
  (layer1): Sequential(
    (0): Conv2d(1, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
    (1): LeakyReLU(negative_slope=0.2, inplace)
  )
  (layer2): Sequential(
    (0): Conv2d(64, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
    (1): InstanceNorm2d(128, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
    (2): LeakyReLU(negative_slope=0.2, inplace)
  )
  (layer3): Sequential(
    (0): Conv2d(128, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
    (1): InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
    (2): LeakyReLU(negative_slope=0.2, inplace)
  )
  (layer4): Sequential(
    (0): Conv2d(256, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
    (1): InstanceNorm2d(512, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
    (2): LeakyReLU(negative_slope=0.2, inplace)
  )
  (output_layer): Sequential(
    (0): ZeroPad2d(padding=(1, 0, 1, 0), value=0.0)
    (1): Conv2d(512, 1, kernel_size=(4, 4), stride=(1, 1), padding=(1, 1), bias=False)
  )
)
```

```python
model_D :
 MNISTDiscriminator(
  (conv_layer): Sequential(
    (0): Conv2d(1, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
    (1): LeakyReLU(negative_slope=0.2, inplace)
    (2): Conv2d(64, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
    (3): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (4): LeakyReLU(negative_slope=0.2, inplace)
  )
  (fc_layer): Sequential(
    (0): Linear(in_features=6272, out_features=1024, bias=True)
    (1): BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): LeakyReLU(negative_slope=0.2)
    (3): Linear(in_features=1024, out_features=1, bias=True)
    (4): Sigmoid()
  )
)
```