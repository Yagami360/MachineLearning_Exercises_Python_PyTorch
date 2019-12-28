# GAN_WGAN_PyTorch
WGAN の PyTorch での実装。

- 参考コード
  - [martinarjovsky/WassersteinGAN](https://github.com/martinarjovsky/WassersteinGAN)

## ■ 項目 [Contents]
1. [動作環境](#動作環境)
1. [使用法](#使用法)
1. [コードの実行結果](#コードの実行結果)
1. [背景理論](https://github.com/Yagami360/My_NoteBook/blob/master/%E6%83%85%E5%A0%B1%E5%B7%A5%E5%AD%A6/%E6%83%85%E5%A0%B1%E5%B7%A5%E5%AD%A6_%E6%A9%9F%E6%A2%B0%E5%AD%A6%E7%BF%92_%E7%94%9F%E6%88%90%E3%83%A2%E3%83%87%E3%83%AB.md#WGAN)

## ■ 動作環境

- Mac OS / ubuntu server
- Python : 3.6
- Anaconda : 5.0.1
- PyTorch : 1.1.0

## ■ 使用法

- 学習処理
  ```sh
  # （例１） train WGAN for MNIST datset using GPU0
  $ python train.py \
    --exper_name WGAN_train \
    --dataset mnist --image_size 64
  ```

- 推論処理（実装中）
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

### ◎ 生成器からの生成画像


### ◎ 損失関数のグラフ

- 識別器 : 1 ~ 50 Epoches<br>
  ![image](https://user-images.githubusercontent.com/25688193/71543178-c9380d00-29b3-11ea-8896-f63121305dfe.png)<br>

- 生成器 : 1 ~ 50 Epoches<br>
  ![image](https://user-images.githubusercontent.com/25688193/71543194-0a302180-29b4-11ea-97cc-67020e5add25.png)<br>

### ◎ 各種オプション引数の設定値

```python
開始時間： 2019-12-27 16:12:19.411065
PyTorch version : 1.1.0
exper_name: WGAN_train_D_vanilla_Opt_RMSprop_Epoch50_191228
device: gpu
dataset: mnist
dataset_dir: ../dataset
results_dir: results
save_checkpoints_dir: checkpoints
load_checkpoints_dir: 
tensorboard_dir: ../tensorboard
n_test: 10000
n_epoches: 50
batch_size: 64
batch_size_test: 256
optimizer: RMSprop
lr_G: 5e-05
lr_D: 5e-05
beta1: 0.5
beta2: 0.999
image_size: 64
n_fmaps: 64
n_input_noize_z: 100
networkD_type: vanilla
n_critic: 5
w_clamp_upper: 0.01
w_clamp_lower: -0.01
n_display_step: 5
n_display_test_step: 100
n_save_step: 10000
seed: 8
debug: True
実行デバイス : cuda
GPU名 : Tesla M60
torch.cuda.current_device() = 0
```


## ■ デバッグ情報
