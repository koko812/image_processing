#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VGG16 で Pascal VOC の元画像をそのまま分類（ImageNet 1000クラス）
- 画像IDを変えたい場合は IMG_ID を変更
- macOS + Apple Silicon なら MPS を自動使用（なければCPU）
"""

from pathlib import Path
from PIL import Image
import torch
from torchvision.models import vgg16, VGG16_Weights

# ==== パス（あなたの例に合わせる）====
BASE = Path("/Users/koko/.cache/kagglehub/datasets/zaraks/pascal-voc-2007/versions/1") \
    / "VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007"
IMG_ID = "000017"  # ← ここを変える
IMG_PATH = BASE / f"JPEGImages/{IMG_ID}.jpg"

def main():
    # デバイス選択（MPS優先→CUDA→CPU）
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print("device:", device)

    # 学習済みVGG16 & 推奨前処理
    weights = VGG16_Weights.IMAGENET1K_V1
    model = vgg16(weights=weights).to(device).eval()
    preprocess = weights.transforms()  # Resize/ToTensor/Normalize をまとめて取得
    idx2label = weights.meta["categories"]

    # 画像読み込み＆前処理
    img = Image.open(IMG_PATH).convert("RGB")
    x = preprocess(img).unsqueeze(0).to(device)

    # 推論
    with torch.no_grad():
        logits = model(x)
        prob = torch.softmax(logits[0], dim=0)

    # 上位5件
    topk = torch.topk(prob, k=5)
    print(f"\nImage: {IMG_PATH}")
    for i in range(5):
        label = idx2label[topk.indices[i].item()]
        score = topk.values[i].item()
        print(f"Top{i+1:>2}: {label:>30s}  {score:.3f}")

if __name__ == "__main__":
    main()

