#!bin/sh
set -eu

# imagenet21k pre-train
mkdir -p checkpoints/imagenet21k
cd checkpoints/imagenet21k

MODEL_NAMES=("ViT-B_16" "ViT-B_32" "ViT-L_16" "ViT-L_32" "ViT-H_14" "R50+ViT-B_16" "R26+ViT-B_32" "R50+ViT-L_32" )
for MODEL_NAME in "${MODEL_NAMES[@]}"
do
    rm -rf ${MODEL_NAME}.npz
    wget https://storage.googleapis.com/vit_models/imagenet21k/${MODEL_NAME}.npz
done

# imagenet21k pre-train + imagenet2012 fine-tuning
mkdir -p checkpoints/imagenet21k_imagenet2012
cd checkpoints/imagenet21k_imagenet2012

MODEL_NAMES=("ViT-B_16-224" "ViT-B_16" "ViT-B_32" "ViT-L_16-224" "ViT-L_16" "ViT-L_32" "R50+ViT-B_16")
for MODEL_NAME in "${MODEL_NAMES[@]}"
do
    rm -rf ${MODEL_NAME}.npz
    wget https://storage.googleapis.com/vit_models/imagenet21k+imagenet2012/${MODEL_NAME}.npz
done
