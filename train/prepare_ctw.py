from pathlib import Path
import cv2
import json
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import jsonlines
# import settings


def each_char(image_anno):
    for block in image_anno['annotations']:
        for char in block:
            yield char


def parse_jsonl_file(img_path, jsonl_path):
    img_path = Path(img_path)
    img = load_image(img_path)
    with jsonlines.open(jsonl_path) as f:
        for line in f.iter():
            if line["file_name"] == img_path.name:
                break
    return img, line
img, line = parse_jsonl_file(
    img_path="D:/ctw-trainval-01-of-26/1001797.jpg",
    jsonl_path="D:/ctw-annotations/train.jsonl"
)
show_image(img)




plt.figure(figsize=(10, 10))
ax = plt.gca()
plt.imshow(img)
for instance in each_char(line):
    color = (0, 1, 0) if instance['is_chinese'] else (1, 0, 0)
    ax.add_patch(
        patches.Polygon(instance['polygon'], fill=False, color=color)
    )
plt.show()
