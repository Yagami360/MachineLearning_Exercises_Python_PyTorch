#!bin/sh
set -eu

mkdir -p checkpoints
cd checkpoints

MODEL_NAMES=("ViT-B_16" "ViT-B_32" "ViT-L_16" "ViT-L_32" "ViT-H_14" )
for MODEL_NAME in "${MODEL_NAMES[@]}"
do
    # imagenet21k pre-train
    wget https://storage.googleapis.com/vit_models/imagenet21k/${MODEL_NAME}.npz
done

<<COMMENTOUT
MODEL_NAMES=("ViT-B_16" "ViT-B_32" "ViT-L_16" "ViT-L_32" "ViT-H_14" )
for MODEL_NAME in "${MODEL_NAMES[@]}"
do
    # imagenet21k pre-train + imagenet2012 fine-tuning
    wget https://storage.googleapis.com/vit_models/imagenet21k+imagenet2012/{MODEL_NAME}.npz
done
COMMENTOUT
