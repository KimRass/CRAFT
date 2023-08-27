import numpy as np
from pathlib import Path
import torch
import torchvision.transforms as T
import cv2

from train_craft.models.craft import CRAFT
from train_craft.torch_utils import _get_state_dict
from image_utils import _get_width_and_height


def load_craft_checkpoint(cuda=False):
    craft = CRAFT()
    if cuda:
        craft = craft.to("cuda")

    ckpt_path = Path(__file__).parent/"pretrained/craft_mlt_25k.pth"
    # ckpt_path = "/Users/jongbeomkim/Desktop/workspace/craft/train_craft/pretrained/craft_mlt_25k.pth"
    # ckpt_path = "D:/craft_mlt_25k.pth"
    state_dict = _get_state_dict(
        ckpt_path=ckpt_path,
        include="module.",
        delete="module.",
        cuda=cuda
    )
    craft.load_state_dict(state_dict=state_dict, strict=True)

    print(f"Loaded pre-trained parameters for 'CRAFT'\n    from checkpoint '{ckpt_path}'.")
    return craft


def _convert_to_uint8(score_map):
    score_map = np.clip(a=score_map, a_min=0, a_max=1)
    score_map *= 255
    score_map = score_map.astype("uint8")
    return score_map


def _resize_image_for_craft_input(img):
    ### Resize the image so that the width and the height are multiples of 32 each. ###
    width, height = _get_width_and_height(img)

    height32, width32 = height, width
    if height % 32 != 0:
        height32 = height + (32 - height % 32)
    if width % 32 != 0:
        width32 = width + (32 - width % 32)

    canvas = np.zeros(shape=(height32, width32, img.shape[2]), dtype=np.uint8)
    resized_img = cv2.resize(src=img, dsize=(width, height), interpolation=cv2.INTER_LANCZOS4)
    canvas[: height, : width, :] = resized_img
    return canvas


def _infer_using_craft(img, craft, cuda=False):
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    z = transform(img)
    z = z.unsqueeze(0)
    if cuda:
        z = z.to("cuda")

    craft.eval()
    with torch.no_grad():
        z, feature = craft(z)
    return z, feature


def _postprocess_score_map(z, ori_width, ori_height, resized_width, resized_height):
    resized_z = cv2.resize(src=z, dsize=(resized_width, resized_height))
    resized_z = resized_z[: ori_height, : ori_width]
    score_map = _convert_to_uint8(resized_z)
    return score_map


def _infer(img, craft, cuda=False):
    z, _ = _infer_using_craft(img=img, craft=craft, cuda=cuda)
    z0 = z[0, :, :, 0].detach().cpu().numpy()
    z1 = z[0, :, :, 1].detach().cpu().numpy()
    ori_width, ori_height = _get_width_and_height(img)
    pred_region = _postprocess_score_map(
        z=z0, ori_width=ori_width, ori_height=ori_height, resized_width=ori_width, resized_height=ori_height
    )
    pred_affinity = _postprocess_score_map(
        z=z1, ori_width=ori_width, ori_height=ori_height, resized_width=ori_width, resized_height=ori_height
    )
    return pred_region, pred_affinity
