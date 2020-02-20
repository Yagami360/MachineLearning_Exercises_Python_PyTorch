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

- Ubuntu : 16.04
- Python : 3.6
- Anaconda : 5.0.1
- PyTorch : 1.x 系
- tensorboardx :
- tqdm :
- imageio :

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

- 学習処理
  ```sh
  # （例１） train PGGAN for MNIST datset using GPU0
  $ python train.py \
    --exper_name PGGAN_train \
    --dataset mnist --init_image_size 4 --final_image_size 32
  ```

- 推論処理（学習済みモデルから画像生成）
  ```sh
  $ python test.py --load_checkpoints_dir ${LOAD_CHAECKPOINTS_DIR}
  ```

- 推論処理（学習済みモデルからモーフィング動画の作成）
  ```sh
  # モーフィング動画の作成
  $ python test_morphing.py --load_checkpoints_dir ${LOAD_CHAECKPOINTS_DIR}
  ```
  ```sh
  # （例）
  $ python test_morphing.py \
      --exper_name PGGAN_test_morphing \
      --load_checkpoints_dir checkpoints/PGGAN_train_Epoch100_191230 \
      --dataset mnist --init_image_size 4 --final_image_size 32 \
      --fps 30 --codec gif
  ```

- TensorBoard
  ```sh
  $ tensorboard --logdir ${TENSOR_BOARD_DIR} --port ${AVAILABLE_POOT}
  ```

  ```sh
  #（例）
  $ tensorboard --logdir tensorboard --port 6006
  ```
  
<a id="コード説明＆実行結果"></a>

## ■ コードの実行結果

### ◎ 生成画像

- [Traing Progress 1] : 4 × 4 pixel 縦（8 枚）×横（８枚）<br>
  - Epoch 0<br>
    ![fake_image_epoches0_batchAll](https://user-images.githubusercontent.com/25688193/71587820-95e7b080-2b62-11ea-98db-6d0925ad3b08.png)

- [Traing Progress 2] : 8 × 8 pixel 縦（8 枚）×横（８枚）<br>
  - Epoch 1<br>
    ![fake_image_epoches1_batchAll](https://user-images.githubusercontent.com/25688193/71587819-95e7b080-2b62-11ea-929c-c327908f0844.png)

- [Traing Progress 3] : 16 × 16 pixel 縦（8 枚）×横（８枚）<br>
  - Epoch 2<br>
    ![fake_image_epoches2_batchAll](https://user-images.githubusercontent.com/25688193/71587817-954f1a00-2b62-11ea-8d62-aa24af0207f1.png)
  - Epoch 3<br>
    ![fake_image_epoches3_batchAll](https://user-images.githubusercontent.com/25688193/71587816-954f1a00-2b62-11ea-9638-de690a81aa11.png)

- [Traing Progress 4] : 32 × 32 pixel 縦（8 枚）×横（８枚）<br>
  - Epoch 4<br>
    ![fake_image_epoches4_batchAll](https://user-images.githubusercontent.com/25688193/71587814-954f1a00-2b62-11ea-8040-479620aa6c9e.png)
  - Epoch 5<br>
    ![fake_image_epoches5_batchAll](https://user-images.githubusercontent.com/25688193/71587813-954f1a00-2b62-11ea-8d6d-5849456a0655.png)
  - Epoch 10<br>
    ![fake_image_epoches10_batchAll](https://user-images.githubusercontent.com/25688193/71587812-94b68380-2b62-11ea-89ad-9d7571b40e8e.png)
  - Epoch 50<br>
    ![fake_image_epoches50_batchAll](https://user-images.githubusercontent.com/25688193/71611900-0254c500-2be0-11ea-9b96-f09f96e09c68.png)<br>

- 学習済みモデルからのモーフィング動画<br>
  ![morphing_video](https://user-images.githubusercontent.com/25688193/71776293-4f66db80-2fd2-11ea-8a91-afc2d57e6b8e.gif)<br>

### ◎ 損失関数のグラフ

- 識別器 : Epoch 1 ~ 50<br>
  ![image](https://user-images.githubusercontent.com/25688193/71611926-30d2a000-2be0-11ea-91d3-799111063391.png)
  - 茶色 : 学習用データセット（ミニバッチ単位）
  - 水色 : テスト用データセット（データセット全体）

- 生成器 : Epoch 1 ~ 50<br>
  ![image](https://user-images.githubusercontent.com/25688193/71611958-58c20380-2be0-11ea-86aa-74edf7eb0dc8.png)<br>
  - 茶色 : 学習用データセット（ミニバッチ単位）
  - 水色 : テスト用データセット（データセット全体）

## ■ デバッグ情報

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