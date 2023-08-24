# Source: https://rrc.cvc.uab.es/?ch=8&com=downloads
import sys
sys.path.insert(0, "/Users/jongbeomkim/Desktop/workspace/craft")

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
        # transform = T.Compose([
        #     T.ToTensor(),
        #     T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # ])
        # image = transform(image)
        
        label_path = self.data_dir/f"""ch8_training_localization_transcription_gt_v2/gt_img_{idx}.txt"""
        with open(label_path, mode="r") as f:
            lines = f.readlines()
        dst_quads = list()
        texts = list()
        for line in lines:
            x1, y1, x2, y2, x3, y3, x4, y4, lang, text = line.strip().split(",")
            dst_quads.append([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
            texts.append(text)
        return {
            "image": image,
            "quadrilaterals": np.array(dst_quads).astype("float32"),
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
    dst_quads = batch["quadrilaterals"]
    # image.show()
    # vis = draw_quads(image=image, quads=quads)
    # vis.show()

    gauss = _get_2d_isotropic_gaussian_map()
    w = h = 200
    margin = 0.1
    xmin = w * margin
    ymin = h * margin
    xmax = w * (1 - margin)
    ymax = h * (1 - margin)
    src_quad = np.array(
        [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]], dtype="float32"
    )
    
    img = np.array(image)
    show_image(img)
    for src_quad in dst_quads:
        dst_quad = _get_dst_quad(src_quad)
        M = cv2.getPerspectiveTransform(src=src_quad, dst=dst_quad)
        w = round(dst_quad[2, 0])
        h = round(dst_quad[2, 1])
        output = cv2.warpPerspective(src=img, M=M, dsize=(w, h))
        show_image(output)
    
    # _, h, w = image.shape
    w, h = image.size
    gt = np.zeros(shape=(h, w))
    for dst_quad in dst_quads:
        M = cv2.getPerspectiveTransform(src=src_quad, dst=dst_quad)
        output = cv2.warpPerspective(src=gauss, M=M, dsize=(w, h))
        gt = np.maximum(gt, output)
    vis_score_map(image=image, score_map=gt)
    
        model.eval()
        _, h, w = transformed.shape
        with torch.no_grad():
            z, feat = model(transformed.unsqueeze(0))
        z0 = z[0, :, :, 0].detach().cpu().numpy()
        resized_z0 = cv2.resize(z0, dsize=(w, h))
        region_map = _normalize_score_map(resized_z0)
        # Image.fromarray(region_map).show()
