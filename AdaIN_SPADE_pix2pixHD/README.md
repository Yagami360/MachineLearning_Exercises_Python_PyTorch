# AdaIN_pix2pixHD
pix2pix-HD の生成器をベースにした場合の AdaIN 入力有無での品質比較

## 各 decoder 層での入力なし
pix2pix-HD 生成器に入力部のみにセグメンテーション画像を入力した場合

- DAなし / epoch 50<br>
    - train データ<br>
        ![image](https://user-images.githubusercontent.com/25688193/97776697-aa933b80-1bad-11eb-9d27-9cef2599a3b4.png)<br>
    - valid データ<br>
        ![image](https://user-images.githubusercontent.com/25688193/97776706-c0a0fc00-1bad-11eb-809f-ea385de03812.png)<br>


## 各 decoder 層での非 AdaIN での入力あり
pix2pix-HD 生成器に入力部と各 decoder 層に 非 AdaIN でセグメンテーション画像を入力した場合

- DAなし / epoch 100

## 各 decoder 層での AdaIN での入力あり（pix2pix-HD の生成器）
pix2pix-HD 生成器の入力部と各 decoder 層にセグメンテーション画像の AdaIN 入力を行った場合

- DAなし / epoch 50
