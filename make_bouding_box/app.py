# app.py
import streamlit as st
from pathlib import Path
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw, ImageFont

# ========= あなたの実装を統合 =========
def parse_voc(xml_path: Path):
    r = ET.parse(xml_path).getroot()
    out = []
    for obj in r.findall("object"):
        name = obj.findtext("name")
        diff = int(obj.findtext("difficult", "0"))
        trunc = int(obj.findtext("truncated", "0"))
        bb = obj.find("bndbox")
        xmin, ymin = int(bb.findtext("xmin")), int(bb.findtext("ymin"))
        xmax, ymax = int(bb.findtext("xmax")), int(bb.findtext("ymax"))
        out.append({
            "name": name, "difficult": diff, "truncated": trunc,
            "bbox": (xmin, ymin, xmax, ymax)
        })
    return out

def color_for(diff: int, trunc: int):
    if diff and trunc:   return "magenta"
    if diff:             return "orange"
    if trunc:            return "blue"
    return "red"

def draw_legend(draw: ImageDraw.ImageDraw, x: int, y: int, font):
    items = [("red", "normal"), ("orange", "difficult"),
             ("blue", "truncated"), ("magenta", "diff+trunc")]
    pad, h = 6, 18
    for i, (c, label) in enumerate(items):
        yy = y + i*(h+4)
        draw.rectangle([x, yy, x+h, yy+h], outline=c, width=3)
        draw.text((x+h+8, yy), label, fill=c, font=font)


def draw_colored(img_path: Path, xml_path: Path) -> Image.Image:
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    # --- フォント設定 ---
    try:
        # 環境に合わせてフォントパスを調整してください
        font = ImageFont.truetype("Arial.ttf", 18)   # サイズ18pt
        legend_font = ImageFont.truetype("Arial.ttf", 16)
    except OSError:
        # フォントが見つからない場合はデフォルトをfallback
        font = ImageFont.load_default()
        legend_font = font

    for o in parse_voc(xml_path):
        x1, y1, x2, y2 = o["bbox"]
        c = color_for(o["difficult"], o["truncated"])
        draw.rectangle([x1, y1, x2, y2], outline=c, width=3)
        label = o["name"]
        if o["difficult"]:  label += " (diff)"
        if o["truncated"]:  label += " (trunc)"
        draw.text((x1, max(0, y1-18)), label, fill=c, font=font)

    # 凡例も少し大きめのフォントで
    draw_legend(draw, 8, 8, legend_font)
    return img


# ---- データセットのベースパス ----
BASE = Path("/Users/koko/.cache/kagglehub/datasets/zaraks/pascal-voc-2007/versions/1")

# JPEG & Annotations パス
JPEG_CANDIDATES = [
    BASE / "VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/JPEGImages",
    BASE / "VOCtest_06-Nov-2007/VOCdevkit/VOC2007/JPEGImages",
]
ANN_CANDIDATES = [
    BASE / "VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/Annotations",
    BASE / "VOCtest_06-Nov-2007/VOCdevkit/VOC2007/Annotations",
]

# JPGを収集
files = []
for p in JPEG_CANDIDATES:
    if p.exists():
        files += sorted([f for f in p.glob("*.jpg")])
if not files:
    st.error("JPEGImages が見つかりませんでした。パスを確認してください。")
    st.stop()

# ---- UI ----
st.title("PASCAL VOC 2007 Viewer (Original vs Colored BBox)")
choice = st.selectbox("画像を選択", [f.name for f in files], index=0)

# ---- 表示処理 ----
path = next(f for f in files if f.name == choice)
img = Image.open(path).convert("RGB")

# 対応する XML
stem = path.stem
ann_path = None
for p in ANN_CANDIDATES:
    candidate = p / f"{stem}.xml"
    if candidate.exists():
        ann_path = candidate
        break

# bbox付き画像を作成
img_with_bb = None
if ann_path:
    img_with_bb = draw_colored(path, ann_path)

# ---- 並べて表示 ----
col1, col2 = st.columns(2)
with col1:
    st.image(img, caption="Original", use_container_width=True)
with col2:
    if img_with_bb:
        st.image(img_with_bb, caption="With Colored BBox", use_container_width=True)
    else:
        st.warning("対応する Annotation (XML) が見つかりませんでした。")

