# UNet_PyTorch
UNet の PyTorch での実装<br>
UNet によるセマンティックセグメンテーションを利用して、衛星画像から地図を生成する。<br>

- 参考コード
    - [GitHub/GunhoChoi/Kind-PyTorch-Tutorial12_Semantic_Segmentation/](https://github.com/GunhoChoi/Kind-PyTorch-Tutorial/tree/master/12_Semantic_Segmentation)

## ■ 項目 [Contents]
1. [動作環境](#動作環境)
1. [使用法](#使用法)
1. [コードの実行結果](#コードの実行結果)
1. [背景理論](https://github.com/Yagami360/My_NoteBook/blob/master/%E6%83%85%E5%A0%B1%E5%B7%A5%E5%AD%A6/%E6%83%85%E5%A0%B1%E5%B7%A5%E5%AD%A6_%E6%A9%9F%E6%A2%B0%E5%AD%A6%E7%BF%92_%E7%94%9F%E6%88%90%E3%83%A2%E3%83%87%E3%83%AB.md#UNet)

## ■ 動作環境

- Mac OS / ubuntu server
- Python : 3.6
- Anaconda : 5.0.1
- PyTorch : 1.1.0
- tensorboard : 1.13.1
- tensorboardx : 1.9
- tqdm

## ■ 使用法

- 使用データ（航空写真と地図画像）<br>
  https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/maps.tar.gz<br>
  よりダウンロード後、解凍。

- 学習処理
  ```sh
  # （例２） train UNet for air map datset using GPU0
  # when save datset dataset/maps dir from https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/maps.tar.gz
  $ python train.py \
    --exper_name UNet_train \
    --dataset_dir dataset/maps \
    --image_size 256
  ```

- 推論処理
  ```sh
  $ python test.py \
    --exper_name UNet_test \
    --dataset_dir ${DATASET_DIR} \
    --load_checkpoints_dir ${LOAD_CHECKPOINTS_DIR} \
    --image_size 64 \
  ```
  ```sh
  # （例）
  $ python test.py \
    --exper_name UNet_test \
    --dataset_dir ../dataset/maps \
    --load_checkpoints_dir checkpoints/UNet_train_Epoch100_191230 \
    --image_size 64
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

### ◎ 生成画像

- Epoches : 10（テストデータ）<br>
  ![fake_image_epoches10_batchAll](https://user-images.githubusercontent.com/25688193/71588082-86b53280-2b63-11ea-8562-04bc2faccd70.png)<br>
- Epoches : 50（テストデータ）<br>
  ![fake_image_epoches50_batchAll](https://user-images.githubusercontent.com/25688193/71588081-86b53280-2b63-11ea-85a4-a22b392fea6c.png)<br>
- Epoches : 75（テストデータ）<br>
  ![fake_image_epoches75_batchAll](https://user-images.githubusercontent.com/25688193/71588197-f7f4e580-2b63-11ea-9eb8-b7a9d773ed88.png)<br>
- Epoches : 100（テストデータ）<br>
  ![fake_image_epoches99_batchAll](https://user-images.githubusercontent.com/25688193/71588080-86b53280-2b63-11ea-9d7f-b080db5417e1.png)<br>
- Epoches 1 ~ 100（テストデータ）<br>
  ![fake_image_epoches99](https://user-images.githubusercontent.com/25688193/71588079-861c9c00-2b63-11ea-895e-0c017f5ab5ea.gif)

→ UNet による生成画像は、その Encoder-Decoder 構造故に、全体的にぼやけた画像担っている点に注目。

### ◎ 損失関数のグラフ

![image](https://user-images.githubusercontent.com/25688193/71588144-c4b25680-2b63-11ea-877c-2aebc6ea3dc8.png)
- 灰色：学習用データセット（ミニバッチ単位）
- オレンジ：テスト用データセット（データセット全体）

### ◎ 各種オプション引数の設定値
```python
開始時間： 2019-12-30 11:43:45.172679
PyTorch version : 1.1.0
exper_name: UNet_train_Epoch100_191230
device: gpu
dataset_dir: ../dataset/maps
results_dir: results
save_checkpoints_dir: checkpoints
load_checkpoints_dir: 
tensorboard_dir: ../tensorboard
n_test: 10000
n_epoches: 100
batch_size: 32
batch_size_test: 64
lr: 0.0002
beta1: 0.5
beta2: 0.999
image_size: 64
n_fmaps: 64
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
model :
 UNet(
  (conv1): Sequential(
    (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): LeakyReLU(negative_slope=0.2, inplace)
    (3): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (4): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  )
  (pool1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (conv2): Sequential(
    (0): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): LeakyReLU(negative_slope=0.2, inplace)
    (3): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (4): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  )
  (pool2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (conv3): Sequential(
    (0): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): LeakyReLU(negative_slope=0.2, inplace)
    (3): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (4): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  )
  (pool3): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (conv4): Sequential(
    (0): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): LeakyReLU(negative_slope=0.2, inplace)
    (3): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (4): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  )
  (pool4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (bridge): Sequential(
    (0): Conv2d(512, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): LeakyReLU(negative_slope=0.2, inplace)
    (3): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (4): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  )
  (dconv1): Sequential(
    (0): ConvTranspose2d(1024, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1))
    (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): LeakyReLU(negative_slope=0.2, inplace)
  )
  (up1): Sequential(
    (0): Conv2d(1024, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): LeakyReLU(negative_slope=0.2, inplace)
    (3): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (4): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  )
  (dconv2): Sequential(
    (0): ConvTranspose2d(512, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1))
    (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): LeakyReLU(negative_slope=0.2, inplace)
  )
  (up2): Sequential(
    (0): Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): LeakyReLU(negative_slope=0.2, inplace)
    (3): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (4): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  )
  (dconv3): Sequential(
    (0): ConvTranspose2d(256, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1))
    (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): LeakyReLU(negative_slope=0.2, inplace)
  )
  (up3): Sequential(
    (0): Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): LeakyReLU(negative_slope=0.2, inplace)
    (3): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (4): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  )
  (dconv4): Sequential(
    (0): ConvTranspose2d(128, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1))
    (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): LeakyReLU(negative_slope=0.2, inplace)
  )
  (up4): Sequential(
    (0): Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): LeakyReLU(negative_slope=0.2, inplace)
    (3): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (4): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  )
  (out_layer): Sequential(
    (0): Conv2d(64, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): Tanh()
  )
)
```