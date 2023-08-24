import torch
from torchvision.utils import make_grid
import torchvision.transforms.functional as TF
from einops import rearrange
from PIL import Image, ImageDraw
import numpy as np
import cv2
from time import time
from datetime import timedelta
from collections import OrderedDict


def _pad_input_image(image):
    """
    Resize the image so that the width and the height are multiples of 16 each.
    """
    _, _, h, w = image.shape
    if h % 16 != 0:
        new_h = h + (16 - h % 16)
    if w % 16 != 0:
        new_w = w + (16 - w % 16)
    new_image = TF.pad(image, padding=(0, 0, new_w - w, new_h - h), padding_mode="constant")
    return new_image


def draw_quads(image, quads):
    copied = image.copy()
    draw = ImageDraw.Draw(copied)
    for quad in quads:
        draw.polygon(xy=quad, outline=(255, 0, 0))
    return copied


def get_elapsed_time(time_start):
    return timedelta(seconds=round(time() - time_start))


def load_ckpt(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    new_ckpt = OrderedDict()
    for old_key in list(ckpt.keys()):
        if old_key and old_key.startswith("module."):
            new_key = old_key.split("module.")[1]
            new_ckpt[new_key] = ckpt[old_key]
    return new_ckpt


def vis_craft_output(z):
    z = rearrange(z, pattern="b h w c -> (b c) h w").unsqueeze(1)
    grid = make_grid(z, nrow=2, normalize=True, value_range=(0, 1))
    image = TF.to_pil_image(grid)
    return image


def _normalize_score_map(score_map):
    score_map = np.clip(a=score_map, a_min=0, a_max=1)
    score_map *= 255
    score_map = score_map.astype(np.uint8)
    return score_map


def _postprocess_score_map(z, ori_width, ori_height, resized_width, resized_height):
    resized_z = cv2.resize(src=z, dsize=(resized_width, resized_height))
    resized_z = resized_z[: ori_height, : ori_width]
    score_map = _normalize_score_map(resized_z)
    return score_map


def _apply_jet_colormap(img):
    img_jet = cv2.applyColorMap(src=(255 - img), colormap=cv2.COLORMAP_JET)
    return img_jet


def _get_l2_dist(p1, p2):
    return np.linalg.norm(p1 - p2, ord=2)


def _get_dst_quad(src_quad):
    w1 = _get_l2_dist(p1=src_quad[0], p2=src_quad[1])
    w2 = _get_l2_dist(p1=src_quad[2], p2=src_quad[3])
    h1 = _get_l2_dist(p1=src_quad[1], p2=src_quad[2])
    h2 = _get_l2_dist(p1=src_quad[3], p2=src_quad[0])
    w = (w1 + w2) // 2
    h = (h1 + h2) // 2
    dst_quad = np.array(((0, 0), (w, 0), (w, h), (0, h)), dtype="float32")
    return dst_quad


# def _get_dst_quad(src_quad):
#     # src_quad = dst_quad.copy()
#     x1, y1, x2, y2, x3, y3, x4, y4 = sum(src_quad, [])
#     w1 = round(((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5)
#     w2 = round(((x3 - x4) ** 2 + (y3 - y4) ** 2) ** 0.5)
#     h1 = round(((x2 - x3) ** 2 + (y2 - y3) ** 2) ** 0.5)
#     h2 = round(((x4 - x1) ** 2 + (y4 - y1) ** 2) ** 0.5)
#     w = (w1 + w2) // 2
#     h = (h1 + h2) // 2
#     dst_quad = [[0, 0], [w, 0], [w, h], [0, h]]
#     return dst_quad


# def _transform_image(image, src_quad):
#     dst_quad = _get_dst_quad(src_quad)
#     new_image = TF.perspective(image, src_quad, dst_quad)
#     new_image = new_image[:, : dst_quad[2][1], : dst_quad[2][0]]
#     return new_image


def _get_2d_isotropic_gaussian_map(w=200, h=200, sigma=0.4):
    x, y = np.meshgrid(
        np.linspace(-1, 1, w), np.linspace(-1, 1, h)
    )
    d = np.sqrt(x ** 2 + y ** 2)
    mu = 0
    gaussian_map = np.exp(
        -(d - mu) ** 2 / (2 * sigma ** 2)
    )

    # gaussian_map *= 255
    # gaussian_map = gaussian_map.astype("uint8")
    return gaussian_map


def _get_gaussian_map_core_rectangle(gaussian_map, margin=0.4):
    w, h = _get_width_and_height(gaussian_map)
    xmin = w * margin
    ymin = h * margin
    xmax = w * (1 - margin)
    ymax = h * (1 - margin)
    return xmin, ymin, xmax, ymax


def _to_pil(img, mode="RGB"):
    if not isinstance(img, Image.Image):
        image = Image.fromarray(img, mode=mode)
        return image
    else:
        return img


def show_image(image):
    _to_pil(image).show()


def show_blended_image(image1, image2, alpha=0.5):
    image1 = _to_pil(image1)
    image2 = _to_pil(image2)
    blended = Image.blend(image1, image2, alpha=alpha)
    blended.show()


def _apply_jet_colormap(img):
    img_jet = cv2.applyColorMap(src=(255 - img), colormap=cv2.COLORMAP_JET)
    return img_jet


def vis_score_map(image, score_map):
    score_map *= 255
    score_map = score_map.astype("uint8")
    score_map = _apply_jet_colormap(score_map)
    show_blended_image(image, score_map)
