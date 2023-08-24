import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np
import pandas as pd
import requests
from io import BytesIO
from pathlib import Path
import random
import math
import cv2

from utils import draw_bboxes, pos_pixel_mask_to_pil
from infer import _pad_input_image

np.set_printoptions(edgeitems=20, linewidth=220, suppress=False)
torch.set_printoptions(precision=4, edgeitems=12, linewidth=220)

IMG_SIZE = 512


def _get_2d_isotropic_gaussian_map(w=200, h=200, sigma=0.4):
    x, y = np.meshgrid(
        np.linspace(-1, 1, w), np.linspace(-1, 1, h)
    )
    d = np.sqrt(x ** 2 + y ** 2)
    mu = 0
    gaussian_map = np.exp(
        -(d - mu) ** 2 / (2 * sigma ** 2)
    )

    gaussian_map *= 255
    gaussian_map = gaussian_map.astype("uint8")
    return gaussian_map


class MenuImageDataset(Dataset):
    def __init__(self, csv_dir, split="train"):

        self.csv_paths = list(Path(csv_dir).glob("*.csv"))
        self.split = split

    def _get_quads(self, csv_path):
        # csv_path = "/Users/jongbeomkim/Desktop/workspace/text_segmenter/data/1037_2300.csv"
        bboxes = pd.read_csv(csv_path, usecols=["xmin", "ymin", "xmax", "ymax"])
        # bboxes.rename({"xmin": "x1", "ymin": "y1", "xmax": "x2", "ymax": "y2"}, axis=1, inplace=True)
        quads = np.array([
            [[row.xmin, row.ymin], [row.xmax, row.ymin], [row.xmax, row.ymax], [row.xmin, row.ymax]]
            for row in bboxes.itertuples()
        ]).astype("float32")
        # quads[0].shape
        # [
        #     [[row.xmin, row.ymin], [row.xmax, row.ymin], [row.xmax, row.ymax], [row.xmin, row.ymax]]
        #     for row in bboxes.itertuples()
        # ]
        # quads = [
        #     [[row.xmin, row.ymin], [row.xmax, row.ymin], [row.xmax, row.ymax], [row.xmin, row.ymax]]
        #     for row in bboxes.itertuples()
        # ]
        return quads

    def load_image(self, csv_path):
        bboxes = pd.read_csv(csv_path, usecols=["image_url"])
        img_path = bboxes["image_url"][0]
        image = Image.open(BytesIO(requests.get(img_path).content)).convert("RGB")
        return image

    def __len__(self):
        return len(self.csv_paths)

    def __getitem__(self, idx):
        csv_path = self.csv_paths[idx]

        quads = self._get_quads(csv_path)
        image = self.load_image(csv_path)
        return image, quads


if __name__ == "__main__":
    mu = 0
    sigma = 0.4
    heatmap = np.zeros(shape=(200, 200))
    for k in range(100):
        # heatmap[k: -k, k] = heatmap[k: -k, -k - 1] = heatmap[k, k: -k] = heatmap[-k - 1, k: -k] = math.sqrt(k)
        heatmap[k: -k, k] = heatmap[k: -k, -k - 1] = heatmap[k, k: -k] = heatmap[-k - 1, k: -k] = min(70, k)
    heatmap /= heatmap.max()
    heatmap *= 255
    heatmap = heatmap.astype("uint8")
    show_image(heatmap)


    # w = h = 200
    # gauss = _get_2d_isotropic_gaussian_map(w=w, h=h, sigma=0.4)

    margin = 0.2
    w = h = 200
    xmin = round(w * margin)
    ymin = round(h * margin)
    xmax = round(w * (1 - margin))
    ymax = round(h * (1 - margin))
    src_quad = np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]).astype("float32")

    CSV_DIR = "/Users/jongbeomkim/Desktop/workspace/text_segmenter/data"
    ds = MenuImageDataset(csv_dir=CSV_DIR, split="train")
    image, dst_quads = ds[0]

    w, h = image.size
    output = np.zeros(shape=(h, w), dtype="uint8")
    for dst_quad in dst_quads:
        M = cv2.getPerspectiveTransform(src=src_quad, dst=dst_quad.astype("float32"))
        out = cv2.warpPerspective(src=heatmap, M=M, dsize=(w, h))
        output = np.maximum(output, out)
    # Image.fromarray(output).show()
    show_image(image, output, 0.7)
    show_image(image, output >= 150, 0.7)
