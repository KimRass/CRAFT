import json
import numpy as np
import cv2
from pprint import pprint
from tqdm.auto import tqdm
import torch

from train_craft.process_images import (
    _get_canvas_same_size_as_image,
    _convert_to_2d,
    _get_masked_image,
    _get_minimum_area_bounding_rotated_rectangle,
    convert_polygon_to_mask
)
from train_craft.watershed import (
    _perform_watershed
)
from train_craft.craft_utilities import (
    _infer
)
from train_craft.generate_ground_truths import (
    _get_2d_isotropic_gaussian_map,
    generate_score_map,
    generate_affinity_score_map,
    get_affinity_quadlilateral,
    get_affinity_quadlilaterals
)


def get_confidence_score(gt_length, pred_length):
    conf_score = (
        gt_length - min(gt_length, abs(gt_length - pred_length))
    ) / gt_length
    return conf_score


def _get_largest_overlapping_mask(trg_mask, masks):
    maxim = 0
    largest_mask = _get_canvas_same_size_as_image(trg_mask, black=True)
    for mask in masks:
        overlap_mask = _get_masked_image(img=trg_mask, mask=mask)
        summ = overlap_mask.sum()
        if summ > maxim:
            maxim = summ
            largest_mask = overlap_mask
    return largest_mask


def _get_region_quadlilaterals(img, region_score_map, polys, score_thresh=50):
    # region_score_map=pred_region
    # score_thresh=70
    poly_masks = [convert_polygon_to_mask(img=img, poly=poly) for poly in polys]
    
    region_segmentation_map = _perform_watershed(region_score_map, score_thresh=score_thresh)
    # show_image(region_segmentation_map, img)
    region_quads = list()
    # max_scores = list()
    for label in np.unique(region_segmentation_map):
        if label == 0:
            continue

        pred_region_mask = (region_segmentation_map == label).astype("uint8") * 255
        mask = _get_largest_overlapping_mask(trg_mask=pred_region_mask, masks=poly_masks)
        if mask.sum() != 0:
            quad = _get_minimum_area_bounding_rotated_rectangle(mask)

            # max_score = pred_region[region_segmentation_map == label].max()
            region_quads.append(quad)
            # max_scores.append(max_score)
    return region_quads


def get_pseudo_score_map(img, quads, margin=0.4):
    gaussian_map = _get_2d_isotropic_gaussian_map()
    pseudo_score_map = _get_canvas_same_size_as_image(
        img=_convert_to_2d(img), black=True
    )
    for quad in quads:
        score_map = generate_score_map(img=img, poly=quad, gaussian_map=gaussian_map, margin=margin)
        pseudo_score_map = np.maximum(pseudo_score_map, score_map)
    return pseudo_score_map


if __name__ == "__main__":
    cuda = torch.cuda.is_available()

    interim = load_craft_checkpoint(cuda)

    label_path = "/Users/jongbeomkim/Downloads/train_labels.json"
    # label_path = "D:/train_labels.json"
    with open(label_path, mode="r") as f:
        labels = json.load(f)
        for trg in tqdm(list(labels.keys())):
            # trg = "gt_1871"
            img_path = f"/Users/jongbeomkim/Downloads/train_images/{trg}.jpg"
            # img_path = f"D:/train_images/{trg}.jpg"
            img = load_image(img_path)
            # show_image(img)

            pred_region, pred_affinity = _infer(img=img, craft=interim, cuda=cuda)
            
            polys = [np.array(word["points"]) for word in labels[trg]]
            region_quads = _get_region_quadlilaterals(
                # img=img, region_score_map=pred_region, polys=polys, score_thresh=165
                img=img, region_score_map=pred_region, polys=polys, score_thresh=80
            )
            # dr = draw_polygons(img, region_quads)
            # show_image(dr)
            pseudo_region = get_pseudo_score_map(img=img, quads=region_quads, margin=0.2)
            # show_image(pseudo_region, img)
            # show_image(pred_region, img)
            
            save_image(
                img1=pseudo_region,
                img2=img,
                path=f"/Users/jongbeomkim/Desktop/workspace/craft/train_craft/datasets/icdar2019/sample_pseudo_score_maps/{trg}_pseudo_region_score_map.jpg"
            )
            save_image(
                img1=pred_region,
                img2=img,
                path=f"/Users/jongbeomkim/Desktop/workspace/craft/train_craft/datasets/icdar2019/sample_pseudo_score_maps/{trg}_predicted_region_score_map.jpg"
            )
            
            # a = _get_canvas_same_size_as_image(img=_convert_to_2d(img), black=True)
            # for word in labels[trg]:
            #     word
            #     polys = [np.array(word["points"])]
            # region_quads = _get_region_quadlilaterals(
            #     img=img, region_score_map=pred_region, polys=polys
            # )
            # len(region_quads)
            poly_masks = [convert_polygon_to_mask(img=img, poly=poly) for poly in polys]
            
            [_get_intersection_of_quarliateral(quad) for quad in region_quads]

            affinity_quads = get_affinity_quadlilaterals(region_quads)
            pseudo_affinity = get_pseudo_score_map(img=img, quads=affinity_quads)
            show_image(pseudo_affinity, img)
            # a = np.maximum(a, pseudo_affinity)
            # show_image(r, img)

            save_image(img1=a, img2=img, path=f"/Users/jongbeomkim/Downloads/score_maps/{trg}_pseudo_afiinity.png")

