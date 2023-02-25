import gc
import torch
import torch.nn as nn
import torchvision.transforms as T
from pathlib import Path
from torch.nn import DataParallel
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
# from torchvision.datasets import ImageFolder

from utilities import (
    get_arguments
)
from process_images import (
    load_image
)
from train_craft.datasets.ctw.prepare_ctw import (
    CTWDataset
)
from train_craft.loss import (
    ScoreMapLoss
)
from train_craft.craft_utilities import (
    load_craft_checkpoint
)

def train(data_dir, batch_size=1, cuda=False):
    n_epochs = 2
    lr = 1e-4
    weight_decay = 5e-4
    print_every = 4
    save_every = 4 * 2

    craft = load_craft_checkpoint(cuda)
    optimizer = optim.Adam(params=craft.parameters(), lr=lr, weight_decay=weight_decay)

    # ctw_ds = CTWDataset("/Users/jongbeomkim/Desktop/workspace/craft/train/datasets/ctw/samples")
    ctw_ds = CTWDataset(data_dir)
    ctw_dl = DataLoader(dataset=ctw_ds, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True, pin_memory=True)

    criterion = ScoreMapLoss(ohem=True)

    craft.train()
    for epoch in range(1, n_epochs + 1):
        running_loss = 0
        for step, (img, gt_region, gt_affinity) in enumerate(ctw_dl, start=1):
            if cuda:
                img = img.cuda()
                gt_region = gt_region.cuda()
                gt_affinity = gt_affinity.cuda()

            optimizer.zero_grad()

            out, _ = craft(img)
            pred_region = out[..., 0].detach.unsqueeze(1)
            pred_affinity = out[..., 1].detach.unsqueeze(1)

            loss = criterion(
                gt_region=gt_region,
                pred_region=pred_region,
                gt_affinity=gt_affinity,
                pred_affinity=pred_affinity
            )

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if step % print_every == 0:
                print(f"| Epoch: {epoch} | Step: {step} | Total loss during the last {print_every} steps: {running_loss} |")

                running_loss = 0

            if step % save_every == 0:
                save_dir = Path(__file__).parent/"train_logs"
                save_dir.mkdir(parents=True, exist_ok=True)
                torch.save(obj=craft.state_dict(), f=save_dir/f"test{step}.pth")

if __name__ == "__main__":
    torch.manual_seed(777)

    cuda = torch.cuda.is_available()

    args = get_arguments()

    gc.collect()
    torch.cuda.empty_cache()
    train(data_dir=args.data_dir, batch_size=args.batch_size, cuda=cuda)
