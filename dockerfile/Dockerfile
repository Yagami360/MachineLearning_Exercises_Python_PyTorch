#-----------------------------
# Docker イメージのベースイメージ
#-----------------------------
# CUDA 10.0 for Ubuntu 16.04
FROM nvidia/cuda:10.0-base-ubuntu16.04

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
# 作業ディレクトリの設定
# 各種命令コマンド（RUN, COPY 等）を実行する際のコンテナ内のカレントディレクトリとなる。
# 又、WORKDIR に設定したフォルダが docker run 時の作業起点ディレクトリとなる。
#-----------------------------
WORKDIR /app

#-----------------------------
# Create a non-root user
#-----------------------------
# Create a non-root user and switch to it
RUN adduser --disabled-password --gecos '' --shell /bin/bash user \
    && chown -R user:user /app
RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user
USER user

# All users can use /home/user as their home directory
ENV HOME=/home/user
RUN chmod 777 /home/user

#-----------------------------
# ENV : 環境変数
#-----------------------------
# 日本語対応
ENV PYTHONIOENCODING utf-8

# コンテナ内から全ての GPU が確認できるように環境変数 NVIDIA_VISIBLE_DEVICES で指定する。（nvidia-docker 用）
#ENV NVIDIA_VISIBLE_DEVICES all

# utility : nvidia-smi コマンドおよび NVML, compute : CUDA / OpenCL アプリケーション（nvidia-docker 用）
#ENV NVIDIA_DRIVER_CAPABILITIES utility,compute

#-----------------------------
# 追加ライブラリのインストール
#-----------------------------
# miniconda のインストール
RUN curl -so ~/miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-4.5.11-Linux-x86_64.sh \
    && chmod +x ~/miniconda.sh \
    && ~/miniconda.sh -b -p ~/miniconda \
    && rm ~/miniconda.sh
ENV PATH=/home/user/miniconda/bin:$PATH
ENV CONDA_AUTO_UPDATE_CONDA=false

# conda 上で Python 3.6 環境を構築
RUN /home/user/miniconda/bin/conda create -y --name py36 python=3.6.9 \
    && /home/user/miniconda/bin/conda clean -ya
ENV CONDA_DEFAULT_ENV=py36
ENV CONDA_PREFIX=/home/user/miniconda/envs/$CONDA_DEFAULT_ENV
ENV PATH=$CONDA_PREFIX/bin:$PATH
RUN /home/user/miniconda/bin/conda install conda-build=3.18.9=py36_3 \
    && /home/user/miniconda/bin/conda clean -ya

# pytorch のインストール（CUDA 10.0-specific steps）
RUN conda install -y -c pytorch \
    cudatoolkit=10.0 \
    "pytorch=1.2.0=py3.6_cuda10.0.130_cudnn7.6.2_0" \
    "torchvision=0.4.0=py36_cu100" \
    && conda clean -ya

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
WORKDIR ${HOME}/share/MachineLearning_Exercises_Python_PyTorch
