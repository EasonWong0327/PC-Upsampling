#!/bin/bash

# 设置目标文件夹和需要复制的文件数量
#TARGET_DIR="/home/jupyter-eason/project/upsampling/Data/ShapeNet"
TARGET_DIR="/home/jupyter-eason/data/point_cloud/8i/8iVFBv2/soldier/Ply"
NUM_FILES=2000

# 进入目标文件夹
cd "$TARGET_DIR"

# 获取所有文件名并排除子文件夹
FILES=$(ls -1 | grep -vE '^(.*)\/$')

# 随机排列文件名并复制指定数量的文件
shuf -n "$NUM_FILES" <<< "$FILES" | xargs -I {} cp {} /home/jupyter-eason/data/point_cloud/8i/8iVFBv2/soldierless/Ply