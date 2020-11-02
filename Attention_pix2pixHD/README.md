# Attention_pix2pixHD
Attention 構造付き pix2pixHD<br>
ネットワーク内部で Attention マスク画像を生成し、attention 重み付け合成を行う<br>
Attention 用マスク画像の生成は、`nn.Conv2d(hoge,1,...)` で１チャネルの特徴マップにして、`nn.Sigmoid()` で 0.0 ~ 1.0 の範囲に activate するだけで生成できる 

## Attention 構造なしの pix2pix-HD

- 学習用データセットのペア数 20 ペア + DA なし<br>
    - train データ<br>
    - valid データ<br>

## Attention 構造ありの pix2pix-HD

- 学習用データセットのペア数 20 ペア + DA なし<br>
    - train データ<br>
    - valid データ<br>


