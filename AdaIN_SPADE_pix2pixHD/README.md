# AdaIN_SPADE_pix2pixHD
pix2pix-HD の生成器をベースにした場合の AdaIN or SPADE 入力有無での品質比較

## 各 decoder 層での入力なし
pix2pix-HD 生成器に入力部のみにセグメンテーション画像を入力した場合

- 学習用データセット 200 ペア + epoch 200 + DA なし<br>
    - valid データでの生成画像<br>
        <img src="https://user-images.githubusercontent.com/25688193/97796852-61e58c00-1c5a-11eb-8a3e-3b90fa57ab94.png" width="300">

## 各 decoder 層での非 AdaIN or SPADE での入力あり
pix2pix-HD 生成器に入力部と各 decoder 層に 非 AdaIN でセグメンテーション画像を入力した場合

- 学習用データセット 200 ペア + epoch 200 + DA なし<br>
    - train データでの生成画像<br>
    - valid データでの生成画像<br>

## 各 decoder 層での AdaIN での入力あり（pix2pix-HD の生成器）
pix2pix-HD 生成器の入力部と各 decoder 層にセグメンテーション画像の AdaIN 入力を行った場合

- 学習用データセット 200 ペア + epoch 200 + DA なし<br>
    - valid データでの生成画像<br>
        <img src="https://user-images.githubusercontent.com/25688193/97796871-93f6ee00-1c5a-11eb-9f9c-443b5f0dff1f.png" width="300">

## 各 decoder 層での SPADE での入力あり（pix2pix-HD の生成器）
pix2pix-HD 生成器の入力部と各 decoder 層にセグメンテーション画像の SPADE 入力を行った場合

- 学習用データセット 200 ペア + epoch 200 + DA なし<br>
    - valid データでの生成画像<br>
        <img src="https://user-images.githubusercontent.com/25688193/97796886-b8eb6100-1c5a-11eb-986e-2d5098b224d0.png" width="300">


→ AdaIN or SPADE 入力ありのほうが、valid データでの生成画像の品質が高い<br>
→ AdaIN or SPADE 入力ありでは、ネットワークに入力しているセグメンテーション画像をよりうまく伝搬できている<br>
