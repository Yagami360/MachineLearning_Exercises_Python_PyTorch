#-----------------------------
# FROM : Docker イメージのベースイメージ
#-----------------------------
FROM pytorch/pytorch
#FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu16.04

#-----------------------------
# ARG : 変数を定義
# buildする時に変更可能
#-----------------------------
# コンテナ内のディレクトを決めておく
ARG ROOT_DIR=/workspace

#-----------------------------
# RUN : コマンド命令
# ここに記述したコマンドを実行してミドルウェアをインストールし、imageのレイヤーを重ねる
#-----------------------------
# apt-get update : インストール可能なパッケージの「一覧」を更新する。
# apt-get install : インストールを実行する。
# -y : 問い合わせがあった場合はすべて「y」と答える
# python & python3-pipのインストール
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    # imageのサイズを小さくするためにキャッシュ削除
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    # pipのアップデート
    && pip install --upgrade pip

#-----------------------------
# ENV : 環境変数
#-----------------------------
# 日本語対応
ENV PYTHONIOENCODING utf-8

# コンテナ内から全ての GPU が確認できるように環境変数 NVIDIA_VISIBLE_DEVICES で指定する。（nvidia-docker 用）
ENV NVIDIA_VISIBLE_DEVICES all

# utility : nvidia-smi コマンドおよび NVML, compute : CUDA / OpenCL アプリケーション（nvidia-docker 用）
ENV NVIDIA_DRIVER_CAPABILITIES utility,compute

#-----------------------------
# COPY ${コピー元（ホスト側にあるファイル）} ${コピー先（）}
# Dockerfileを実行したディレクトにあるファイルのコピー
#-----------------------------
COPY requirements.txt ${ROOT_DIR}

#-----------------------------
# ライブラリのインストール
#-----------------------------
WORKDIR ${ROOT_DIR}
RUN pip install -r requirements.txt

#-----------------------------
# ディレクトリの移動
#-----------------------------
WORKDIR ${ROOT_DIR}/MachineLearning_Exercises_Python_PyTorch
