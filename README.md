# MachineLearning_Exercises_Python_PyTorch
PyTorch 実装の練習コード集。<br>

## ■ 動作環境
nvidia 製 GPU 搭載マシンでの動作を想定しています。

### ◎ conda 環境
- Ubuntu : 
    - シェルスクリプト `.sh` のみ Ubuntu での動作を想定しています。
- Python : 3.6
- Anaconda :
- PyTorch : 1.x 系
- tensorboardx :
- tqdm :
- imageio :
- Pillow < 7.0.0

### ◎ Docker 環境
nvidia-docker2 で動作します。

- Docker イメージの作成 ＆ Docker コンテナの起動（docker-compose を使用する場合）
    ```sh
    $ docker-compose up -d
    $ docker exec -it -u $(id -u $USER):$(id -g $USER) ml_exercises_pytorch_container /bin/bash
    ```

<!--
- Docker イメージの作成
    ```sh
    $ docker build ./dockerfile -t ml_exercises_pytorch_image
    ```

- Docker コンテナの起動（nvidia-docker2）
    ```sh
    $ docker run -it --rm -v ${PWD}:/home/user/share/MachineLearning_Exercises_Python_PyTorch --name ml_exercises_container ml_exercises_pytorch_image --runtime nvidia --p 6006:6006 /bin/bash
    ```

- docker-compose を使用する場合
    ```sh
    $ docker-compose up -d
    ```
-->

## ■ 項目（フォルダ別）

1. Deep Neural Network（基礎モデル）
    1. [ResNet](https://github.com/Yagami360/MachineLearning_Exercises_Python_PyTorch/tree/master/ResNet_PyTorch)
    1. [UNet](https://github.com/Yagami360/MachineLearning_Exercises_Python_PyTorch/tree/master/UNet_PyTorch)
1. CNN 系
    1. [CNN_GeometricMatchingCNN](https://github.com/Yagami360/MachineLearning_Exercises_Python_PyTorch/tree/master/CNN_GeometricMatchingCNN)
1. GANs
    1. [Deep Convolutional GAN（DCGAN）](https://github.com/Yagami360/MachineLearning_Exercises_Python_PyTorch/tree/master/GAN_DCGAN_PyTorch)
    1. [Conditional GAN（cGAN）](https://github.com/Yagami360/MachineLearning_Exercises_Python_PyTorch/tree/master/GAN_cGAN_PyTorch)
    1. [Wasserstein GAN（WGAN）](https://github.com/Yagami360/MachineLearning_Exercises_Python_PyTorch/tree/master/GAN_WGAN_PyTorch)
    1. [Improved Training of Wasserstein GANs（WGAN-GP）](https://github.com/Yagami360/MachineLearning_Exercises_Python_PyTorch/tree/master/GAN_WGAN-GP_PyTorch)
    1. [Relativistic GANs（RGAN）](https://github.com/Yagami360/MachineLearning_Exercises_Python_PyTorch/tree/master/GAN_RGAN_PyTorch)
    1. [ProgressiveGAN](https://github.com/Yagami360/MachineLearning_Exercises_Python_PyTorch/tree/master/GAN_PGGAN_PyTorch)
    1. [Pix2Pix](https://github.com/Yagami360/MachineLearning_Exercises_Python_PyTorch/tree/master/GAN_Pix2Pix_PyTorch)
    1. [Pix2Pix-HD](https://github.com/Yagami360/MachineLearning_Exercises_Python_PyTorch/tree/master/GAN_Pix2PixHD_PyTorch)
1. Graph Convolutional Networks
    1. [GCN_simple_classication](https://github.com/Yagami360/MachineLearning_Exercises_Python_PyTorch/tree/master/GCN_simple_classication)
    1. [GCN_graphonomy](https://github.com/Yagami360/MachineLearning_Exercises_Python_PyTorch/tree/master/GCN_graphonomy)
1. 正則化層
    1. [AdaIN_SPADE_pix2pixHD](https://github.com/Yagami360/MachineLearning_Exercises_Python_PyTorch/tree/master/AdaIN_SPADE_pix2pixHD)
1. 強化学習
    1. [【外部リンク】ReinforcementLearning](https://github.com/Yagami360/ReinforcementLearning_Exercises)
1. 仮想試着モデル
    1. [【外部リンク】virtual-try-on_exercises_pytorch](https://github.com/Yagami360/virtual-try-on_exercises_pytorch)

## ■ 参考文献＆サイト
