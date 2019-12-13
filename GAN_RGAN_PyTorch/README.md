# GAN_RGAN-GP_PyTorch
RSGAN, RaSGAN, RaLSGAN の PyTorch での実装。

## ■ 項目 [Contents]
1. [動作環境](#動作環境)
1. [使用法](#使用法)
1. [コードの実行結果](#コードの実行結果)
1. [背景理論](https://github.com/Yagami360/MachineLearning-Papers_Survey/issues/51)

## ■ 動作環境

- Mac OS / ubuntu server
- Python : 3.6
- Anaconda : 5.0.1
- PyTorch : 1.1.0

## ■ 使用法

- 学習処理
```sh
$ python train.py
```

- 推論処理
```sh
$ python test.py
```

- TensorBoard
```sh
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

- RSGAN<br>
    ![image](https://user-images.githubusercontent.com/25688193/70800236-1088a000-1def-11ea-9a72-5bb289360563.png)<br>
    ![image](https://user-images.githubusercontent.com/25688193/70800017-7fb1c480-1dee-11ea-86bc-ca0c63a1dc11.png)<br>


- RaSGAN<br>

### ◎ 生成器から生成された自動生成画像

- RSGAN<br>
    - Epochs :10<br>
            ![image](https://user-images.githubusercontent.com/25688193/70800135-d15a4f00-1dee-11ea-9275-0dd44068a599.png)<br>

- RaSGAN<br>
    - Epochs :10<br>
        ![image](https://user-images.githubusercontent.com/25688193/70803801-42523480-1df8-11ea-8807-4dfa0149fe4b.png)<br>


## ■ デバッグ情報
