- Reference: https://github.com/backtime92/CRAFT-Reimplementation/blob/craft/loss/mseloss.py

import torch
import torch.nn as nn


criterion = nn.MSELoss(reduction="sum")
criterion(gt_region, pred_region)