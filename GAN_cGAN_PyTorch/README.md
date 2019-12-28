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

### ◎ 損失関数のグラフ

- 識別器 : Epoches 1~100<br>
  ![image](https://user-images.githubusercontent.com/25688193/71542919-f387cb80-29af-11ea-865e-20d94e511552.png)<br>

- 生成器 : Epoches 1~100
  ![image](https://user-images.githubusercontent.com/25688193/71542934-3ea1de80-29b0-11ea-8dec-952a676526ac.png)<br>

  → 学習を進めていくとむしろ悪化？

## ■ デバッグ情報

```python
[CGAN Generator]
```

```python

```
