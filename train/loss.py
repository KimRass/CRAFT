- References: https://github.com/backtime92/CRAFT-Reimplementation/blob/craft/loss/mseloss.py, https://arxiv.org/pdf/1604.03540.pdf

import numpy as np
import torch
import torch.nn as nn


class ScoreMapLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def single_image_loss(self, loss, gt):
        batch_size = loss.shape[0]
        # sum_loss = torch.mean(loss.view(-1))*0
        # loss = loss.view(batch_size, -1)
        # gt = gt.view(batch_size, -1)

        positive_pixel = (gt > 0.1).float()
        positive_pixel_number = torch.sum(positive_pixel)
        positive_loss_region = loss * positive_pixel
        positive_loss = torch.sum(positive_loss_region) / positive_pixel_number

        negative_pixel = (gt <= 0.1).float()
        negative_pixel_number = torch.sum(negative_pixel)

        if negative_pixel_number < 3 * positive_pixel_number:
            negative_loss_region = loss * negative_pixel
            negative_loss = torch.sum(negative_loss_region) / negative_pixel_number
        else:
            negative_loss_region = loss * negative_pixel
            negative_loss = torch.sum(
                torch.topk(negative_loss_region.view(-1), int(3*positive_pixel_number))[0]
            ) / (positive_pixel_number * 3)

        # negative_loss_region = loss * negative_pixel
        # negative_loss = torch.sum(negative_loss_region) / negative_pixel_number

        total_loss = positive_loss + negative_loss
        return total_loss

    def forward(self, gt_region, gt_affinity, pred_region, pred_affinity, confidence_map):
        loss_fn = nn.MSELoss(reduction="none")

        assert gt_region.size() == pred_region.size() and gt_affinity.size() == pred_affinity.size()
        region_loss = loss_fn(pred_region, gt_region)
        affinity_loss = loss_fn(pred_affinity, gt_affinity)
        region_loss = torch.mul(region_loss, confidence_map)
        affinity_loss = torch.mul(affinity_loss, confidence_map)

        region_loss = self.single_image_loss(loss=region_loss, gt=gt_region)
        affinity_loss = self.single_image_loss(loss=affinity_loss, gt=gt_affinity)
        return region_loss + affinity_loss
