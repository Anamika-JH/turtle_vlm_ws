#!/usr/bin/env bash
set -e

# you can change DEST to another directory if you like
DEST="$(dirname "$0")"
mkdir -p "$DEST" && cd "$DEST"

echo "↓ YOLOv8x checkpoint ..."
wget -q --show-progress -O yolov8x.pt \
  https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8x.pt

echo "↓ Segment-Anything ViT-B checkpoint ..."
wget -q --show-progress -O sam_vit_b_01ec64.pth \
  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

echo "All models are in $DEST"
