#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pascal VOC の XML を読み込んで情報を表示する最小スクリプト
"""

import xml.etree.ElementTree as ET
from pathlib import Path
import sys

def parse_voc_xml(xml_path: Path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find("size")
    width = int(size.findtext("width"))
    height = int(size.findtext("height"))
    depth = int(size.findtext("depth", default="3"))

    print(f"[{xml_path.name}]")
    print(f"  image size: {width}x{height}, depth={depth}")

    for i, obj in enumerate(root.findall("object"), start=1):
        name = obj.findtext("name")
        difficult = int(obj.findtext("difficult", default="0"))
        truncated = int(obj.findtext("truncated", default="0"))
        bb = obj.find("bndbox")
        xmin = int(bb.findtext("xmin"))
        ymin = int(bb.findtext("ymin"))
        xmax = int(bb.findtext("xmax"))
        ymax = int(bb.findtext("ymax"))
        print(f"  object {i}: {name} "
              f"(bbox=({xmin},{ymin},{xmax},{ymax}), "
              f"difficult={difficult}, truncated={truncated})")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("使い方: python voc_xml_parse.py <xml file>")
        sys.exit(1)

    xml_file = Path(sys.argv[1])
    parse_voc_xml(xml_file)

