# Processing_PyTorch
PyTorch での処理フローの練習コード集。（※機械学習のコードではありません。）<br>

## ■ 項目 [Contents]
1. [動作環境](#動作環境)
1. [使用法](#使用法)

## ■ 動作環境

- Python : 3.6
- Anaconda : 5.0.1
- PyTorch : 1.0.0
- scikit-learn : 0.20.2

## ■ 使用法

- 使用法
```
$ python main.py
```

## ■ Tips

### 自動微分

- `detach()` の効果
- 

### ネットワーク定義

- `nn.Sequential()` を用いたネットワーク定義

- `nn.ModuleList()` を用いたネットワーク定義

- ネットワークの名前付け
    - `setattr()` を用いて、`nn.Sequential()` に動的に名前付けを行う。
