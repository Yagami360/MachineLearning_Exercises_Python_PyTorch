# MachineLearning_Exercises_Python_PyTorch
PyTorch 実装の練習コード集。<br>

## ■ 動作環境

- Ubuntu : 16.04
- Python : 3.6
- Anaconda : 5.0.1
- PyTorch : 1.x 系
- tensorboardx :
- tqdm :
- imageio :

### ◎ Docker 環境で動かす場合
nvidia-docker2 での動作を想定しています。

- Docker イメージの作成 ＆ Docker コンテナの起動（docker-compose を使用する場合）
    ```sh
    $ docker-compose up -d
    $ docker exec -it ml_exercises_pytorch_container /bin/bash
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
1. GANs
    1. [Deep Convolutional GAN（DCGAN）](https://github.com/Yagami360/MachineLearning_Exercises_Python_PyTorch/tree/master/GAN_DCGAN_PyTorch)
    1. [Conditional GAN（cGAN）](https://github.com/Yagami360/MachineLearning_Exercises_Python_PyTorch/tree/master/GAN_cGAN_PyTorch)
    1. [Wasserstein GAN（WGAN）](https://github.com/Yagami360/MachineLearning_Exercises_Python_PyTorch/tree/master/GAN_WGAN_PyTorch)
    1. [Improved Training of Wasserstein GANs（WGAN-GP）](https://github.com/Yagami360/MachineLearning_Exercises_Python_PyTorch/tree/master/GAN_WGAN-GP_PyTorch)
    1. [Relativistic GANs（RGAN）](https://github.com/Yagami360/MachineLearning_Exercises_Python_PyTorch/tree/master/GAN_RGAN_PyTorch)
    1. [ProgressiveGAN](https://github.com/Yagami360/MachineLearning_Exercises_Python_PyTorch/tree/master/GAN_PGGAN_PyTorch)
    1. [Pix2Pix](https://github.com/Yagami360/MachineLearning_Exercises_Python_PyTorch/tree/master/GAN_Pix2Pix_PyTorch)
    1. [Pix2Pix-HD](https://github.com/Yagami360/MachineLearning_Exercises_Python_PyTorch/tree/master/GAN_Pix2PixHD_PyTorch)
- 強化学習
    1. [【外部リンク】ReinforcementLearning](https://github.com/Yagami360/ReinforcementLearning_Exercises)
- 仮想試着モデル
    1. [【外部リンク】virtual-try-on_exercises_pytorch](https://github.com/Yagami360/virtual-try-on_exercises_pytorch)

## ■ 参考文献＆サイト
