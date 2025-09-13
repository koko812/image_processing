#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pascal VOC の XML を読み、元画像と bbox 描画画像を保存する
保存先: image_with_bounding_box/<id>/without_bd.png, with_bd.png
"""

import xml.etree.ElementTree as ET
from pathlib import Path
from PIL import Image, ImageDraw

def parse_voc_xml(xml_path: Path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    objs = []
    for obj in root.findall("object"):
        name = obj.findtext("name")
        bb = obj.find("bndbox")
        xmin = int(bb.findtext("xmin"))
        ymin = int(bb.findtext("ymin"))
        xmax = int(bb.findtext("xmax"))
        ymax = int(bb.findtext("ymax"))
        objs.append({"name": name, "bbox": (xmin, ymin, xmax, ymax)})
    return objs

def save_with_without_bbox(img_path: Path, xml_path: Path, out_dir: Path):
    img_id = img_path.stem
    # 出力フォルダ作成
    save_dir = out_dir / img_id
    save_dir.mkdir(parents=True, exist_ok=True)

    # 元画像をコピー保存
    img = Image.open(img_path).convert("RGB")
    img.save(save_dir / "without_bd.png")

    # bbox 描画版
    img_bbox = img.copy()
    draw = ImageDraw.Draw(img_bbox)
    objs = parse_voc_xml(xml_path)
    for obj in objs:
        bbox = obj["bbox"]
        cls = obj["name"]
        draw.rectangle(bbox, outline="red", width=3)
        draw.text((bbox[0], bbox[1]-10), cls, fill="red")
    img_bbox.save(save_dir / "with_bd.png")

if __name__ == "__main__":
    base = Path("/Users/koko/.cache/kagglehub/datasets/zaraks/pascal-voc-2007/versions/1/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007")
    img_path = base / "JPEGImages/000012.jpg"
    xml_path = base / "Annotations/000012.xml"
    out_dir = Path("image_with_bounding_box")

    save_with_without_bbox(img_path, xml_path, out_dir)
    print("保存完了:", out_dir / img_path.stem)

