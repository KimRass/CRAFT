import torch
import torch.nn as nn
import torchvision.transforms as T
from pathlib import Path
from torch.nn import DataParallel
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
# from torchvision.datasets import ImageFolder

from process_images import (
    load_image
)
from datasets.ctw.prepare_ctw import (
    CTWDataset
)


# gt_region = region_score_map
# pred_region = affinity_score_map

# criterion = nn.MSELoss(reduction="none")
# # criterion = nn.MSELoss(reduce=False, size_average=False)
# criterion(gt_region, pred_region)





# craft = CRAFT()
craft = load_craft_checkpoint(lang="ko", cuda=False)
# craft = DataParallel(craft).cuda()
lr = 1e-4
weight_decay = 5e-4
optim.Adam(params=craft.parameters(), lr=lr, weight_decay=weight_decay)




ctw_ds = CTWDataset(data_dir="D:/ctw_out")
ctw_dl = DataLoader(dataset=ctw_ds, batch_size=4, shuffle=True, num_workers=0)
for batch, (img, region_score_map, affinity_score_map) in enumerate(ctw_dl):
    img.shape, region_score_map.shape
    
region_score_map = _reverse_jet_colormap(
    load_image("D:/ctw_out/0000172_region.png")
)
