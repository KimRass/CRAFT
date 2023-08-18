# References: https://github.com/backtime92/CRAFT-Reimplementation/blob/craft/loss/mseloss.py, https://arxiv.org/pdf/1604.03540.pdf

import torch
import torch.nn as nn


class ScoreMapLoss(nn.Module):
    def __init__(self, ohem=True):
        super().__init__()
        self.ohem = ohem

    def _get_total_loss_using_ohem(self, loss_map, gt):
        is_pos_pixel = (gt > 0.1).float()
        n_pos_pixels = int(torch.sum(is_pos_pixel))
        pos_loss_map = loss_map * is_pos_pixel

        is_neg_pixel = 1. - is_pos_pixel
        n_neg_pixels = int(torch.sum(is_neg_pixel))
        neg_loss_map = loss_map * is_neg_pixel

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

    def forward(self, gt_region, pred_region, gt_affinity, pred_affinity, confidence_map=None):
        criterion = nn.MSELoss(reduction="none")

        region_loss_map = criterion(gt_region, pred_region)
        affinity_loss_map = criterion(gt_affinity, pred_affinity)
        if confidence_map is not None:
            region_loss_map = torch.mul(region_loss_map, confidence_map)
            affinity_loss_map = torch.mul(affinity_loss_map, confidence_map)

        if self.ohem:
            tot_region_loss = self._get_total_loss_using_ohem(loss_map=region_loss_map, gt=gt_region)
            tot_affinity_loss = self._get_total_loss_using_ohem(loss_map=affinity_loss_map, gt=gt_affinity)
        else:
            tot_region_loss = torch.sum(region_loss_map)
            tot_affinity_loss = torch.sum(affinity_loss_map)
        return tot_region_loss + tot_affinity_loss


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