# Source: https://rrc.cvc.uab.es/?ch=8&com=downloads

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from einops import rearrange
from PIL import Image, ImageDraw
import cv2
from pathlib import Path
from torchvision.utils import make_grid
import torchvision.transforms.functional as TF
import numpy as np

import config
from utils import draw_quads
from model import CRAFT
from utils import load_ckpt, vis_craft_output, _normalize_score_map, _postprocess_score_map


class CIDAR2017(Dataset):
    def __init__(self, data_dir):
        super().__init__()

        self.data_dir = Path(data_dir)

    def __len__(self):
        return 7200
        # return len(
        #     [i for i in list((self.data_dir).glob("**/*")) if "ch8_training_images_" in str(i.parent)]
        # )

    def __getitem__(self, idx):
        img_stem = self.data_dir/f"""ch8_training_images_{(idx // 1000 + 1)}/img_{idx}"""
        img_path = img_stem.with_suffix(".jpg")
        if not img_path.exists():
            img_path.with_suffix(".png")
        image = Image.open(img_path).convert("RGB")
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        image = transform(image)
        
        label_path = self.data_dir/f"""ch8_training_localization_transcription_gt_v2/gt_img_{idx}.txt"""
        with open(label_path, mode="r") as f:
            lines = f.readlines()
        quads = list()
        texts = list()
        for line in lines:
            x1, y1, x2, y2, x3, y3, x4, y4, lang, text = line.strip().split(",")
            # label.append((x1, y1, x2, y2, x3, y3, x4, y4, text))
            # quads.append(tuple(map(int, [x1, y1, x2, y2, x3, y3, x4, y4])))
            quads.append(
                [[int(x1), int(y1)], [int(x2), int(y2)], [int(x3), int(y3)], [int(x4), int(y4)]]
            )
            texts.append(text)
        return {
            "image": image,
            "quadrilaterals": quads,
            "texts": texts,
        }


if __name__ == "__main__":
    model = CRAFT(pretrained=True)
    ckpt_path = "/Users/jongbeomkim/Downloads/craft_mlt_25k.pth"
    ckpt = load_ckpt(ckpt_path)
    model.load_state_dict(ckpt)

    data_dir = "/Users/jongbeomkim/Documents/datasets/icdar2017/"
    ds = CIDAR2017(data_dir)
    batch = ds[127]
    image = batch["image"]
    quads = batch["quadrilaterals"]
    # image.show()
    # vis = draw_quads(image=image, quads=quads)
    # vis.show()

    for src_quad in quads:
        transformed = _transform_image(image, src_quad)
        transformed.shape

        model.eval()
        _, h, w = transformed.shape
        with torch.no_grad():
            z, feat = model(transformed.unsqueeze(0))
        z0 = z[0, :, :, 0].detach().cpu().numpy()
        resized_z0 = cv2.resize(z0, dsize=(w, h))
        region_map = _normalize_score_map(resized_z0)
        Image.fromarray(region_map).show()
