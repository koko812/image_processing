#!/usr/bin/env python3
from pathlib import Path
from PIL import Image

BASE = Path("/Users/koko/.cache/kagglehub/datasets/zaraks/pascal-voc-2007/versions/1") \
       / "VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007"
OUT  = Path("image_with_segmentation")
ALPHA = 0.5  # 透過率（0〜1）

def overlay_mask(img_id: str, mask_kind: str):
    """
    mask_kind: 'class' or 'object'
    出力: image_with_segmentation/<id>/overlay_<kind>.png
    """
    jpg = BASE / "JPEGImages" / f"{img_id}.jpg"
    if mask_kind == "class":
        mask = BASE / "SegmentationClass" / f"{img_id}.png"
    else:
        mask = BASE / "SegmentationObject" / f"{img_id}.png"

    if not jpg.exists() or not mask.exists():
        return False

    out_dir = OUT / img_id
    out_dir.mkdir(parents=True, exist_ok=True)

    img  = Image.open(jpg).convert("RGB")
    m_rgba = Image.open(mask).convert("RGBA")

    # 透明度を一定値に（元のアルファは無視）
    r, g, b, _ = m_rgba.split()
    a = Image.new("L", img.size, int(255 * ALPHA))
    m_rgba = Image.merge("RGBA", (r, g, b, a))

    over = img.copy()
    over.paste(m_rgba, (0, 0), m_rgba)

    # 保存（最初に一度だけ元画像も保存）
    (out_dir / "original.jpg").exists() or img.save(out_dir / "original.jpg")
    over.save(out_dir / f"overlay_{mask_kind}.png")
    return True

if __name__ == "__main__":
    img_id = "000032"  # ここを好きなIDに
    ok_cls = overlay_mask(img_id, "class")
    ok_obj = overlay_mask(img_id, "object")
    if not (ok_cls or ok_obj):
        print(f"No segmentation mask for {img_id}")
    else:
        print("saved to:", (OUT / img_id))

