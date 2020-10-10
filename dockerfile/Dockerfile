#-----------------------------
# Docker イメージのベースイメージ
#-----------------------------
# CUDA 10.1 for Ubuntu 16.04
FROM nvidia/cuda:10.1-base-ubuntu16.04

#-----------------------------
# 基本ライブラリのインストール
#-----------------------------
# apt-get update : インストール可能なパッケージの「一覧」を更新する。
# apt-get install : インストールを実行する。
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
    # imageのサイズを小さくするためにキャッシュ削除
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

#-----------------------------
# ENV : 環境変数
#-----------------------------
ENV LC_ALL=C.UTF-8
ENV export LANG=C.UTF-8
ENV PYTHONIOENCODING utf-8

#-----------------------------
# 追加ライブラリのインストール
#-----------------------------
# miniconda のインストール
RUN curl -LO http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN bash Miniconda3-latest-Linux-x86_64.sh -p /miniconda -b
RUN rm Miniconda3-latest-Linux-x86_64.sh
ENV PATH=/miniconda/bin:${PATH}
RUN conda update -y conda
    
# conda 上で Python 3.6 環境を構築
ENV CONDA_DEFAULT_ENV=py36
RUN conda create -y --name ${CONDA_DEFAULT_ENV} python=3.6.9 && conda clean -ya
ENV CONDA_PREFIX=/miniconda/envs/${CONDA_DEFAULT_ENV}
ENV PATH=${CONDA_PREFIX}/bin:${PATH}
RUN conda install conda-build=3.18.9=py36_3 && conda clean -ya

# pytorch 1.4 のインストール（CUDA 10.1-specific steps）
RUN conda install -y pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=10.1 -c pytorch && conda clean -ya    

# OpenCV3 のインストール
RUN sudo apt-get update && sudo apt-get install -y --no-install-recommends \
    libgtk2.0-0 \
    libcanberra-gtk-module \
    && sudo rm -rf /var/lib/apt/lists/*

RUN conda install -y -c menpo opencv3=3.1.0 && conda clean -ya

# tensorflow のインストール
RUN conda install -y tensorboard && conda clean -ya

# Others
RUN conda install -y tqdm && conda clean -ya
RUN conda install -y imageio && conda clean -ya
RUN conda install -y -c conda-forge tensorboardx && conda clean -ya
RUN conda install -y -c anaconda pillow==6.2.1 && conda clean -ya

#-----------------------------
# コンテナ起動後に自動的に実行するコマンド
#-----------------------------
#CMD ["/bin/bash"]

#-----------------------------
# コンテナ起動後の作業ディレクトリ
#-----------------------------
WORKDIR /MachineLearning_Exercises_Python_PyTorch
