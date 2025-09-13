# app.py
import streamlit as st
from pathlib import Path
from PIL import Image

# ここをあなたの環境に合わせてセット
BASE = Path("/Users/koko/.cache/kagglehub/datasets/zaraks/pascal-voc-2007/versions/1")

# 典型パス（trainval/test のどちらにも JPEGImages がある）
CANDIDATES = [
    BASE / "VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/JPEGImages",
    BASE / "VOCtest_06-Nov-2007/VOCdevkit/VOC2007/JPEGImages",
]

# JPGを収集
files = []
for p in CANDIDATES:
    if p.exists():
        files += sorted([f for f in p.glob("*.jpg")])
if not files:
    st.error("JPEGImages が見つかりませんでした。パスを確認してください。")
    st.stop()

st.title("PASCAL VOC 2007 Viewer (select & show)")
choice = st.selectbox("画像を選択", [f.name for f in files], index=0)

# 表示
path = next(f for f in files if f.name == choice)
img = Image.open(path)
st.image(img, caption=str(path), use_container_width=True)

