# compare_full_vs_crop.py
from pathlib import Path
from PIL import Image
import xml.etree.ElementTree as ET
import torch
from torchvision.models import vgg16, VGG16_Weights

BASE = Path("/Users/koko/.cache/kagglehub/datasets/zaraks/pascal-voc-2007/versions/1") \
    / "VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007"
IMG_ID = "000017"
IMG_PATH = BASE / f"JPEGImages/{IMG_ID}.jpg"
XML_PATH = BASE / f"Annotations/{IMG_ID}.xml"

def first_bbox(xml_path: Path):
    r = ET.parse(xml_path).getroot()
    obj = r.find("object")
    bb = obj.find("bndbox")
    x1 = int(bb.findtext("xmin")); y1 = int(bb.findtext("ymin"))
    x2 = int(bb.findtext("xmax")); y2 = int(bb.findtext("ymax"))
    return (x1,y1,x2,y2)

def top1(model, preprocess, img):
    x = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        p = torch.softmax(model(x)[0], dim=0)
    v, i = torch.topk(p, k=1)
    return v.item(), i.item()

# device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# model & preprocess
weights = VGG16_Weights.IMAGENET1K_V1
model = vgg16(weights=weights).to(device).eval()
preprocess = weights.transforms()
idx2label = weights.meta["categories"]

# full image
img = Image.open(IMG_PATH).convert("RGB")
v_full, i_full = top1(model, preprocess, img)

# crop (1つ目のbboxだけ)
x1,y1,x2,y2 = first_bbox(XML_PATH)
crop = img.crop((x1,y1,x2,y2))
v_crop, i_crop = top1(model, preprocess, crop)

print("Image:", IMG_PATH)
print(f"FULL : {idx2label[i_full]:>25s}  prob={v_full:.3f}")
print(f"CROP : {idx2label[i_crop]:>25s}  prob={v_crop:.3f}")
print(f"Δprob = {v_crop - v_full:+.3f}")

