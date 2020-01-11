# Processing_PyTorch
PyTorch 実装の Tips 集


## ■ 目次
1. 自動微分関連
1. ネットワーク定義関連

## ■ 自動微分関連

- `detach()` による勾配計算の無効化
- `for param in model_D.parameters(): param.requires_grad = False` による勾配計算の無効化
- `with torch.no_grad():` による勾配計算の無効化
- `model.eval()` の効果

## ■ ネットワーク定義関連

- `nn.Sequential()` を用いたネットワーク定義

- `nn.ModuleList()` を用いたネットワーク定義

- ネットワークの名前付け
    - `setattr()` を用いて、`nn.Sequential()` に動的に名前付けを行う。

- ネットワークを End2End にする
