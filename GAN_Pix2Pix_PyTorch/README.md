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

## ■ 使用法

- 使用データ（航空写真と地図画像）<br>
  https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/maps.tar.gz<br>
  よりダウンロード後、解凍。

- 学習処理
  ```sh
  # （例） train Pix2Pix for air map datset using GPU0
  # when save datset dataset/maps dir from https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/maps.tar.gz
  $ python train.py \
    --exper_name Pix2Pix_train \
    --dataset_dir dataset/maps \
    --image_size 64 \
    --unetG_dropout 0.5 --networkD_type PatchGAN
  ```

- 推論処理
  ```sh
  $ python test.py \
    --exper_name Pix2Pix_test \
    --dataset_dir ${DATASET_DIR} \
    --load_checkpoints_dir ${LOAD_CHECKPOINTS_DIR} \
    --image_size 64 \
    --unetG_dropout 0.5 --networkD_type PatchGAN
  ```
  ```sh
  # （例）
  $ python test.py \
    --exper_name Pix2Pix_test \
    --dataset_dir dataset/maps \
    --load_checkpoints_dir checkpoints/Pix2Pix_train_D_PatchGAN_Epoch100_191231 \
    --image_size 64 \
    --unetG_dropout 0.5 --networkD_type PatchGAN
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

### ◎ 生成器からの生成画像

- Epoches : 10<br>
  ![fake_image_epoches10_batchAll](https://user-images.githubusercontent.com/25688193/71606211-51840100-2bb2-11ea-8feb-6b3603abce70.png)<br>
- Epoches : 50<br>
  ![fake_image_epoches50_batchAll](https://user-images.githubusercontent.com/25688193/71606207-51840100-2bb2-11ea-82c6-19a80d99d48d.png)<br>
- Epoches : 100<br>
  ![fake_image_epoches99_batchAll](https://user-images.githubusercontent.com/25688193/71606210-51840100-2bb2-11ea-8543-593afadbcc1e.png)<br>
- Epoches : 1 ~ 100<br>
  ![fake_image_epoches99](https://user-images.githubusercontent.com/25688193/71606208-51840100-2bb2-11ea-9ab8-3796264c022e.gif)<br>

→ UNet による生成画像は全体的にぼやけた画像になっていたが、生成器の構造が同じ UNet 構造となっている Pix2Pix では全体的にくっきりとした画像が生成出来ていることに注目。

### ◎ 損失関数のグラフ

- 識別器側 : Epoches 1 ~ 100<br>
  ![image](https://user-images.githubusercontent.com/25688193/71606246-90b25200-2bb2-11ea-9792-e5e2d3e6dfc2.png)<br>
- 生成器側 : Epoches 1 ~ 100<br>
  ![image](https://user-images.githubusercontent.com/25688193/71606259-a758a900-2bb2-11ea-8721-6b2643d59321.png)<br>
  - 青色：学習用データセット（ミニバッチ単位）
  - 茶色：テスト用データセット（テスト用データ全体）

## ■ デバッグ情報

```python
[生成器]
model_G :
 Pix2PixUNetGenerator(
  (conv1): Sequential(
    (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): LeakyReLU(negative_slope=0.2, inplace)
    (3): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (4): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (5): Dropout(p=0.5)
  )
  (pool1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (conv2): Sequential(
    (0): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): LeakyReLU(negative_slope=0.2, inplace)
    (3): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (4): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (5): Dropout(p=0.5)
  )
  (pool2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (conv3): Sequential(
    (0): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): LeakyReLU(negative_slope=0.2, inplace)
    (3): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (4): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (5): Dropout(p=0.5)
  )
  (pool3): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (conv4): Sequential(
    (0): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): LeakyReLU(negative_slope=0.2, inplace)
    (3): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (4): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (5): Dropout(p=0.5)
  )
  (pool4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (bridge): Sequential(
    (0): Conv2d(512, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): LeakyReLU(negative_slope=0.2, inplace)
    (3): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (4): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (5): Dropout(p=0.5)
  )
  (dconv1): Sequential(
    (0): ConvTranspose2d(1024, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1))
    (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): LeakyReLU(negative_slope=0.2, inplace)
    (3): Dropout(p=0.5)
  )
  (up1): Sequential(
    (0): Conv2d(1024, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): LeakyReLU(negative_slope=0.2, inplace)
    (3): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (4): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (5): Dropout(p=0.5)
  )
  (dconv2): Sequential(
    (0): ConvTranspose2d(512, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1))
    (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): LeakyReLU(negative_slope=0.2, inplace)
    (3): Dropout(p=0.5)
  )
  (up2): Sequential(
    (0): Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): LeakyReLU(negative_slope=0.2, inplace)
    (3): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (4): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (5): Dropout(p=0.5)
  )
  (dconv3): Sequential(
    (0): ConvTranspose2d(256, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1))
    (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): LeakyReLU(negative_slope=0.2, inplace)
    (3): Dropout(p=0.5)
  )
  (up3): Sequential(
    (0): Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): LeakyReLU(negative_slope=0.2, inplace)
    (3): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (4): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (5): Dropout(p=0.5)
  )
  (dconv4): Sequential(
    (0): ConvTranspose2d(128, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1))
    (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): LeakyReLU(negative_slope=0.2, inplace)
    (3): Dropout(p=0.5)
  )
  (up4): Sequential(
    (0): Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): LeakyReLU(negative_slope=0.2, inplace)
    (3): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (4): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (5): Dropout(p=0.5)
  )
  (out_layer): Sequential(
    (0): Conv2d(64, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): Tanh()
  )
)
```

```python
[識別器]
model_D :
 Pix2PixPatchGANDiscriminator(
  (layer1): Sequential(
    (0): Conv2d(6, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
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
