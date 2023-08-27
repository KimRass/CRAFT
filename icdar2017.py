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
from utils import (
    load_ckpt,
    vis_craft_output,
    _normalize_score_map,
    _postprocess_score_map,
    _get_dst_quad,
    _pad_input_image,
)


class CIDAR2017(Dataset):
    def __init__(self, data_dir, interim):
        super().__init__()

        self.data_dir = Path(data_dir)

    def __len__(self):
        return 7200
        # return len(
        #     [i for i in list((self.data_dir).glob("**/*")) if "ch8_training_images_" in str(i.parent)]
        # )

    def __getitem__(self, idx):
        img_stem = self.data_dir/f"""ch8_training_images_{((idx - 1) // 1000 + 1)}/img_{idx}"""
        img_path = img_stem.with_suffix(".jpg")
        if not img_path.exists():
            img_path = img_stem.with_suffix(".png")
        image = Image.open(img_path).convert("RGB")
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
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


def _crop_using_src_quad(img, src_quad):
    dst_quad = _get_dst_quad(src_quad)
    M = cv2.getPerspectiveTransform(src=src_quad, dst=dst_quad)
    w = round(dst_quad[2, 0])
    h = round(dst_quad[2, 1])
    output = cv2.warpPerspective(src=img, M=M, dsize=(w, h))
    return output


transfrom = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def _z_to_array(z):
    score_map = np.clip(a=z, a_min=0, a_max=1)
    score_map *= 255
    score_map = score_map.astype("uint8")
    return score_map


def _infer(model, img):
    model.eval()

    h, w, _ = img.shape

    image = transfrom(img)
    image = image.unsqueeze(0)
    image = _pad_input_image(image)
    with torch.no_grad():
        z, _ = model(image)
    z = z.permute(0, 3, 1, 2)
    z = F.interpolate(z, scale_factor=2)
    z = z[..., : h, : w]

    z0 = z.squeeze()[0, ...].detach().cpu().numpy()
    z1 = z.squeeze()[1, ...].detach().cpu().numpy()

    region_map = _z_to_array(z0)
    link_map = _z_to_array(z1)
    return region_map, link_map


if __name__ == "__main__":
    model = CRAFT(pretrained=True)
    ckpt_path = "/Users/jongbeomkim/Downloads/craft_mlt_25k.pth"
    ckpt = load_ckpt(ckpt_path)
    model.load_state_dict(ckpt)

    data_dir = "/Users/jongbeomkim/Documents/datasets/icdar2017/"
    ds = CIDAR2017(data_dir=data_dir, interim=model)
    batch = ds[2000]
    image = batch["image"]
    dst_quads = batch["quadrilaterals"]
    # image.show()
    # vis = draw_quads(image=image, quads=quads)
    # vis.show()
    
    img = np.array(image)
    show_image(img)
    for src_quad in dst_quads:
        src_quad = dst_quads[0]
        patch = _crop_using_src_quad(img=img, src_quad=src_quad)
        
        region_map, link_map = _infer(model=model, img=patch)
        show_blended_image(patch, _apply_jet_colormap(region_map))

    # gauss = _get_2d_isotropic_gaussian_map()
    # w = h = 200
    # margin = 0.1
    # xmin = w * margin
    # ymin = h * margin
    # xmax = w * (1 - margin)
    # ymax = h * (1 - margin)
    # src_quad = np.array(
    #     [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]], dtype="float32"
    # )