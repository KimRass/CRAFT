# References: https://github.com/backtime92/CRAFT-Reimplementation/blob/craft/loss/mseloss.py, https://arxiv.org/pdf/1604.03540.pdf

import torch
import torch.nn as nn


class ScoreMapLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def perform_ohem(self, loss, gt):
        is_pos_pixel = (gt > 0.1).float()
        n_pos_pixels = int(torch.sum(is_pos_pixel))
        pos_loss_map = loss * is_pos_pixel

        is_neg_pixel = 1. - is_pos_pixel
        n_neg_pixels = int(torch.sum(is_neg_pixel))
        neg_loss_map = loss * is_neg_pixel

        tot_pos_loss = torch.sum(pos_loss_map) / n_pos_pixels

        k = 3 * n_pos_pixels
        # If the image has 3 times more background pixels than text pixels,
        if n_neg_pixels >= k:
            # Then pick top `k` highest loss values.
            # tot_neg_loss = torch.sum(torch.topk(neg_loss_map.view(-1), k=k)[0]) / k
            tot_neg_loss = torch.sum(torch.topk(neg_loss_map.view(-1), k=k)[0])
        else:
            # tot_neg_loss = torch.sum(neg_loss_map) / n_neg_pixels
            tot_neg_loss = torch.sum(neg_loss_map)
        # `tot_loss` consistes of 25% of positive pixels and 75% of negative pixels.
        tot_loss = tot_pos_loss + tot_neg_loss
        return tot_loss

    def forward(self, gt_region, pred_region, gt_affinity, pred_affinity, confidence_map=None, ohem=True):
        if ohem:
            criterion = nn.MSELoss(reduction="none")
        else:
            criterion = nn.MSELoss(reduction="sum")

        assert gt_region.size() == pred_region.size() and gt_affinity.size() == pred_affinity.size()
        region_loss = criterion(pred_region, gt_region)
        affinity_loss = criterion(pred_affinity, gt_affinity)
        if confidence_map is not None:
            region_loss = torch.mul(region_loss, confidence_map)
            affinity_loss = torch.mul(affinity_loss, confidence_map)

        if ohem:
            region_loss = self.perform_ohem(loss=region_loss, gt=gt_region)
            affinity_loss = self.perform_ohem(loss=affinity_loss, gt=gt_affinity)
        return region_loss + affinity_loss


if __name__ == "__main__":
    criterion = ScoreMapLoss()
    criterion(
        gt_region=region_score_map,
        pred_region=affinity_score_map,
        gt_affinity=affinity_score_map,
        pred_affinity=affinity_score_map,
        ohem=True
    )
# False: 2439.9019
# True: 26.9967