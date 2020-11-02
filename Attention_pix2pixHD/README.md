# Attention_pix2pixHD
Attention 構造付き pix2pixHD<br>
ネットワーク内部で Attention マスク画像を生成し、attention 重み付け合成を行う<br>
Attention 用マスク画像の生成は、`nn.Conv2d(n_in_channels,1,...)` で１チャネルの特徴マップにして、`nn.Sigmoid()` で 0.0 ~ 1.0 の範囲に activate するだけで生成できる 

<!--
## Attention 構造なしの pix2pix-HD

- 学習用データセットのペア数 4 ペア + epoch 200 + DA なし<br>
    - train データ<br>
    - valid データ<br>

- 学習用データセットのペア数 4 ペア + epoch 200 + DA あり<br>
    - train データ<br>
    - valid データ<br>
-->

## Attention 構造ありの pix2pix-HD

- 学習用データセットのペア数 4 ペア + DA なし<br>
    - train データ<br>
        <img src="https://user-images.githubusercontent.com/25688193/97831433-60c46580-1d13-11eb-938b-910fe741637e.png" width="300"><br>
        <img src="https://user-images.githubusercontent.com/25688193/97831488-85204200-1d13-11eb-8491-e47d43c330f4.png" width="300"><br>

<!--
- 学習用データセットのペア数 4 ペア + epoch 200 + DA あり<br>
    - train データ<br>
    - valid データ<br>
-->

→ 左から｛変換元顔画像（ドメインA）・変換先顔画像の正解（ドメインB）・生成顔画像・attention マスク画像・コンテンツ画像｝<br>
→ 顔の表情の attention mask が生成されている<br>
