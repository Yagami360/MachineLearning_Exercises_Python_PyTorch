# DA_DeepSIM
DeepSIM [Deep Single Image Manipulation] で提案されている TPS 変換でのデータオーギュメントの有効性検証<br>
生成器のネットワークは、pix2pix-HD の生成器で検証<br>

- 論文まとめ
    - https://github.com/Yagami360/MachineLearning-Papers_Survey/issues/107

## DA なし

- 学習用データセットのペア数 1 ペア + epoch 2000 + 識別器 PatchGAN<br>
    - test データ<br>

## affine 変換での DA あり

- 学習用データセットのペア数 1 ペア + epoch 2000 + 識別器 PatchGAN<br>
    - test データ<br>
        <img src="https://user-images.githubusercontent.com/25688193/98370245-a23f7280-207d-11eb-8b6a-132b6a3a60ac.png" width="600">


## affine 変換 + TPS 変換での DA あり

- 学習用データセットのペア数 1 ペア + epoch 2000 + 識別器 PatchGAN<br>
    - test データ<br>
        <img src="https://user-images.githubusercontent.com/25688193/98430877-2f251300-20f4-11eb-99cd-76215676f9d2.png" width="600">

→ TPS 変換での DA を追加することで、生成画像の品質が向上している。