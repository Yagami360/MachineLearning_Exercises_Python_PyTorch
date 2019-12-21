# GAN_cGAN_PyTorch
Conditional GAN（cGAN）の PyTorch での実装。
ネットワーク構成は、CNN を使用（DCGAN or LSGANベース）

## ■ 項目 [Contents]
1. [動作環境](#動作環境)
1. [使用法](#使用法)
1. [コードの実行結果](#コードの実行結果)
1. [背景理論](https://github.com/Yagami360/My_NoteBook/blob/master/%E6%83%85%E5%A0%B1%E5%B7%A5%E5%AD%A6/%E6%83%85%E5%A0%B1%E5%B7%A5%E5%AD%A6_%E6%A9%9F%E6%A2%B0%E5%AD%A6%E7%BF%92_%E7%94%9F%E6%88%90%E3%83%A2%E3%83%87%E3%83%AB.md#ConditionalGAN%EF%BC%88CGAN%EF%BC%89)

## ■ 動作環境

- Mac OS / ubuntu server
- Python : 3.6
- Anaconda : 5.0.1
- PyTorch : 1.1.0

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
[CGAN Generator]
```

```python

```
