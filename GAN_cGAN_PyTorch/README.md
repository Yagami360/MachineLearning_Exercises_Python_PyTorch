# GAN_cGAN_PyTorch
Conditional GAN（cGAN）の PyTorch での実装。
ネットワーク構成は、CNN を使用（DCGAN or LSGANベース）

## ■ 項目 [Contents]
1. [動作環境](#動作環境)
1. [使用法](#使用法)
1. [コードの実行結果](#コードの実行結果)
1. [背景理論](https://github.com/Yagami360/My_NoteBook/blob/master/%E6%83%85%E5%A0%B1%E5%B7%A5%E5%AD%A6/%E6%83%85%E5%A0%B1%E5%B7%A5%E5%AD%A6_%E6%A9%9F%E6%A2%B0%E5%AD%A6%E7%BF%92_%E7%94%9F%E6%88%90%E3%83%A2%E3%83%87%E3%83%AB.md#ConditionalGAN%EF%BC%88CGAN%EF%BC%89)

## ■ 動作環境

- Ubuntu : 16.04
- Python : 3.6
- Anaconda : 5.0.1
- PyTorch : 1.x 系
- tensorboardx :
- tqdm :
- imageio :

## ■ 使用法

- 学習処理
  ```sh
  # （例１） train cGAN for MNIST datset using GPU0 with vanilla GAN
  $ python train.py \
    --exper_name CGAN_gantype_vanilla_train \
    --dataset mnist --image_size 64 --n_classes 10 \
    --gan_type vanilla
  ```

  ```sh
  # （例２） train cGAN for MNIST datset using GPU0 with LSGAN
  $ python train.py \
    --exper_name CGAN_gantype_LSGAN_train \
    --dataset mnist --image_size 64 --n_classes 10 \
    --gan_type LSGAN
  ```

- 推論処理（学習済みモデルから画像生成）
  ```sh
  # （例１）label 0 の画像を生成
  $ python test.py --load_checkpoints_dir ${LOAD_CHAECKPOINTS_DIR} --y_label 0

  # （例１）label 1 の画像を生成
  $ python test.py --load_checkpoints_dir ${LOAD_CHAECKPOINTS_DIR} --y_label 1
  ```

- 推論処理（学習済みモデルからモーフィング動画の作成）
  ```sh
  # モーフィング動画の作成
  $ python test_morphing.py --load_checkpoints_dir ${LOAD_CHAECKPOINTS_DIR} --y_label ${Y_LABEL}
  ```
  ```sh
  # （例１）label 0 のモーフィング動画を生成
  $ python test_morphing.py \
      --exper_name CGAN_test_morphing \
      --load_checkpoints_dir checkpoints/DCGAN_train_G_vanilla_D_vanilla_Epoch100_191227 \
      --y_label 0 \
      --fps 30 --codec gif
  ```
  ```sh
  # （例２）label 1 のモーフィング動画を生成
  $ python test_morphing.py \
      --exper_name CGAN_test_morphing \
      --load_checkpoints_dir checkpoints/DCGAN_train_G_vanilla_D_vanilla_Epoch100_191227 \
      --y_label 1 \
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

<a id="コードの実行結果"></a>

## ■ コードの実行結果

### ◎ 生成器からの生成画像

- label 0 / Epoche 15
  ![fake_image_label0_epoches15_batchAll](https://user-images.githubusercontent.com/25688193/71542944-74df5e00-29b0-11ea-8276-522ecb740f6d.png)<br>

- label 1 / Epoche 15
  ![fake_image_label1_epoches15_batchAll](https://user-images.githubusercontent.com/25688193/71542992-0353df80-29b1-11ea-924b-76374f859ff8.png)<br>

- label 2 / Epoche 15
  ![fake_image_label2_epoches15_batchAll](https://user-images.githubusercontent.com/25688193/71542991-02bb4900-29b1-11ea-9f46-1c1b036d5c83.png)<br>

- label 3 / Epoche 15
  ![fake_image_label3_epoches15_batchAll](https://user-images.githubusercontent.com/25688193/71542990-02bb4900-29b1-11ea-9d49-825b36db25a0.png)<br>

- label 4 / Epoche 15
  ![fake_image_label4_epoches15_batchAll](https://user-images.githubusercontent.com/25688193/71542989-02bb4900-29b1-11ea-948d-6a0458e3cff1.png)<br>

- label 5 / Epoche 15
  ![fake_image_label5_epoches15_batchAll](https://user-images.githubusercontent.com/25688193/71542988-02bb4900-29b1-11ea-9a4b-9dd9565e7fef.png)<br>

- label 6 / Epoche 15
  ![fake_image_label6_epoches15_batchAll](https://user-images.githubusercontent.com/25688193/71542987-0222b280-29b1-11ea-824d-2ddbeb28eb20.png)<br>

- label 7 / Epoche 15
  ![fake_image_label7_epoches15_batchAll](https://user-images.githubusercontent.com/25688193/71542986-0222b280-29b1-11ea-883c-3f9903ec932d.png)<br>

- label 8 / Epoche 15
  ![fake_image_label8_epoches15_batchAll](https://user-images.githubusercontent.com/25688193/71542985-0222b280-29b1-11ea-84dc-185994575076.png)<br>

- label 9 / Epoche 15
  ![fake_image_label9_epoches15_batchAll](https://user-images.githubusercontent.com/25688193/71543012-3f874000-29b1-11ea-873e-f53a309cf60d.png)<br>


#### ☆ 生成器からの生成画像（失敗ケース）

- label 1 / Epoches 50<br>
  ![fake_image_label1_epoches50_batchAll](https://user-images.githubusercontent.com/25688193/71560236-83aa3b80-2aaa-11ea-899b-7f0d42596477.png)<br>
  → 学習を進めすぎると、生成画像が崩壊した。<br>
  → GAN の Adv loss の種類（vanilla）が問題か？

### ◎ 損失関数のグラフ

- 識別器 : Epoches 1~100<br>
  ![image](https://user-images.githubusercontent.com/25688193/71542919-f387cb80-29af-11ea-865e-20d94e511552.png)<br>

- 生成器 : Epoches 1~100
  ![image](https://user-images.githubusercontent.com/25688193/71542934-3ea1de80-29b0-11ea-8dec-952a676526ac.png)<br>

  → 学習を進めていくとむしろ悪化している。
  → GAN の Adv loss の種類 (vanilla) が問題か？

### ◎ 各種オプション引数の設定値

```python
開始時間： 2019-12-27 15:12:51.169508
PyTorch version : 1.1.0
exper_name: CGAN_train_gantype_vanilla_D_vanilla_Epoch100_191227
device: gpu
dataset: mnist
n_classes: 10
dataset_dir: ../dataset
results_dir: results
save_checkpoints_dir: checkpoints
load_checkpoints_dir: 
tensorboard_dir: ../tensorboard
n_test: 10000
n_epoches: 100
batch_size: 64
batch_size_test: 256
lr: 0.0001
beta1: 0.5
beta2: 0.999
image_size: 64
n_fmaps: 64
n_input_noize_z: 100
gan_type: vanilla
networkD_type: vanilla
n_display_step: 10
n_display_test_step: 100
n_save_step: 10000
seed: 8
debug: True
実行デバイス : cuda
GPU名 : Tesla M60
torch.cuda.current_device() = 0
```

## ■ デバッグ情報

```python
model_G :
 CGANGenerator(
  (layer): Sequential(
    (0): ConvTranspose2d(110, 512, kernel_size=(4, 4), stride=(1, 1), bias=False)
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
model_D :
 CGANDiscriminator(
  (layer): Sequential(
    (0): Conv2d(11, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    (1): LeakyReLU(negative_slope=0.2, inplace)
    (2): Conv2d(64, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    (3): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (4): LeakyReLU(negative_slope=0.2, inplace)
    (5): Conv2d(128, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    (6): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (7): LeakyReLU(negative_slope=0.2, inplace)
    (8): Conv2d(256, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    (9): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (10): LeakyReLU(negative_slope=0.2, inplace)
    (11): Conv2d(512, 1, kernel_size=(4, 4), stride=(1, 1), bias=False)
  )
)
```
※ Discriminator の出力層は、損失関数 に `nn.BCEWithLogitsLoss()` を使用しているため、sigmoid による活性化関数はなし。
