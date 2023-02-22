- Reference: https://github.com/backtime92/CRAFT-Reimplementation/blob/craft/loss/mseloss.py

import torch
import torch.nn as nn
import torchvision.transforms as T


gt_region = data["region_score_map"][0]
pred_region = region

criterion = nn.MSELoss(reduction="none")
criterion(T.ToTensor()(gt_region), T.ToTensor()(pred_region))[0][100, 100]

