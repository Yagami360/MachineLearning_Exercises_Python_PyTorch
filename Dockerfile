#-----------------------------
# Docker イメージのベースイメージ
#-----------------------------
FROM pytorch/pytorch
#FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu16.04

#-----------------------------
# 作業ディレクトリの設定
# 各種命令コマンド（RUN, COPY 等）を実行する際のコンテナ内のカレントディレクトリとなる。
# 又、WORKDIR に設定したフォルダが docker run 時の作業起点ディレクトリとなる。
#-----------------------------
ARG WORK_DIR=/workspace
ARG ROOT_DIR=${WORK_DIR}/MachineLearning_Exercises_Python_PyTorch
WORKDIR ${WORK_DIR}

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
# 基本ライブラリのインストール
#-----------------------------
# apt-get update : インストール可能なパッケージの「一覧」を更新する。
# apt-get install : インストールを実行する。
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
# 追加ライブラリのインストール
#-----------------------------
COPY requirements.txt ${WORK_DIR}
RUN pip install -r requirements.txt

#-----------------------------
# コンテナが作成された後で自動雨滴に実行するコマンド
#-----------------------------
CMD ["/bin/bash"]

#-----------------------------
# コンテナが起動後の作業ディレクトリ
#-----------------------------
WORKDIR ${ROOT_DIR}