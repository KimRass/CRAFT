import torch
import torch.nn as nn
from collections import OrderedDict


def _get_state_dict(
    ckpt_path, key="state_dict", include="", delete="", cuda=False
):
    ckpt = torch.load(ckpt_path, map_location="cuda" if cuda else "cpu")

    if key in ckpt:
        state_dict = ckpt[key]
    else:
        state_dict = ckpt

    new_state_dict = OrderedDict()
    for old_key in list(state_dict.keys()):
        if old_key and old_key.startswith(include):
            new_key = old_key[len(delete):]
            new_state_dict[new_key] = state_dict[old_key]
    return new_state_dict


def move_to_device(obj, device):
    if isinstance(obj, nn.Module) or torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, (tuple, list)):
        return [move_to_device(el, device) for el in obj]
    if isinstance(obj, dict):
        return {name: move_to_device(val, device) for name, val in obj.items()}
    raise ValueError(f"Unexpected type {type(obj)}")


def initialize_weights(modules):
    for m in modules:
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()
