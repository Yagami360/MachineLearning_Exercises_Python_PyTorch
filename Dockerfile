# docker build ./ -t ml_exercises_pytorch_image
#-----------------------------
# Docker イメージのベースイメージ
#-----------------------------
# pytorch 環境をベースにする
FROM pytorch/pytorch
#FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu16.04

#-----------------------------
# ARGで変数を定義
# buildする時に変更可能
#-----------------------------
# コンテナ内のディレクトを決めておく
ARG ROOT_DIR=/workspace

#-----------------------------
#
#-----------------------------
# -y : 問い合わせがあった場合はすべて「y」と答える
# python3 pipのインストール
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    # imageのサイズを小さくするためにキャッシュ削除
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    # pipのアップデート
    && pip install --upgrade pip

#-----------------------------
# 環境変数
#-----------------------------
# 日本語対応
ENV PYTHONIOENCODING utf-8

#-----------------------------
# Dockerfileを実行したディレクトにあるファイルのコピー
# COPY ${コピー元（ホスト側にあるファイル）} ${コピー先（）}
#-----------------------------
COPY requirements.txt ${ROOT_DIR}
#COPY requirements.txt /

#-----------------------------
# ライブラリのインストール
#-----------------------------
WORKDIR ${ROOT_DIR}
RUN pip install -r requirements.txt

#-----------------------------
# ディレレクトリの移動
#-----------------------------
WORKDIR ${ROOT_DIR}/MachineLearning_Exercises_Python_PyTorch
