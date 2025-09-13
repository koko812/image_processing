from pathlib import Path
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw, ImageFont

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
        out.append({"name": name, "difficult": diff, "truncated": trunc,
                    "bbox": (xmin, ymin, xmax, ymax)})
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

def save_colored(img_path: Path, xml_path: Path, out_path: Path):
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    for o in parse_voc(xml_path):
        x1, y1, x2, y2 = o["bbox"]
        c = color_for(o["difficult"], o["truncated"])
        draw.rectangle([x1, y1, x2, y2], outline=c, width=3)
        label = o["name"]
        if o["difficult"]:  label += " (diff)"
        if o["truncated"]:  label += " (trunc)"
        draw.text((x1, max(0, y1-12)), label, fill=c, font=font)

    # 凡例
    draw_legend(draw, 8, 8, font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)

if __name__ == "__main__":
    base = Path("/Users/koko/.cache/kagglehub/datasets/zaraks/pascal-voc-2007/versions/1") \
        / "VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007"
    img_id = "000005"
    save_colored(base/"JPEGImages"/f"{img_id}.jpg",
                 base/"Annotations"/f"{img_id}.xml",
                 Path("image_with_bounding_box")/img_id/"with_bd_colored.png")
    print("saved to:", Path("image_with_bounding_box")/img_id/"with_bd_colored.png")

