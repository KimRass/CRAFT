import numpy as np
from pathlib import Path

from train_craft.craft import (
    CRAFT
)
from train_craft.torch_utilities import (
    _get_state_dict
)


def load_craft_checkpoint(cuda=False):
    craft = CRAFT()
    if cuda:
        craft = craft.to("cuda")

    ckpt_path = Path(__file__).parent/"pretrained/craft_mlt_25k.pth"
    # ckpt_path = "/Users/jongbeomkim/Desktop/workspace/image_processing_server/pretrained/craft.pth"
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


# z0 = z[0, ..., 0].detach().cpu().numpy()
# z1 = z[0, ..., 1].detach().cpu().numpy()
# region_score_map = np.clip(a=z0, a_min=0, a_max=1)
# affinity_score_map = np.clip(a=z1, a_min=0, a_max=1)

# region_score_map = _convert_to_uint8(region_score_map)
# show_image(region_score_map)

# temp = region_score_map
# temp *= 255
# temp = temp.astype("uint8")

# region_score_map = load_image("D:/ctw_out/0000172_region.png")
# region_score_map.max()
