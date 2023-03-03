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


def _get_largest_overlapping_mask(mask1, masks):
    maxim = 0
    largest_mask = _get_canvas_same_size_as_image(mask1, black=True)
    for mask2 in masks:
        overlap_mask = _get_masked_image(img=mask1, mask=mask2)
        summ = overlap_mask.sum()
        if summ > maxim:
            maxim = summ
            largest_mask = overlap_mask
    return largest_mask


def _get_region_quadlilaterals(img, region_score_map, labels[trg]):
    poly_masks = list()
    for word in labels[trg]:
        poly=np.array(word["points"])
        polygon_mask = convert_polygon_to_mask(img=img, poly=poly)
        # show_image(polygon_mask, img)
        poly_masks.append(polygon_mask)
    # len(poly_masks)

    # temp = _get_canvas_same_size_as_image(img=_convert_to_2d(img), black=True)
    region_segmentation_map = _perform_watershed(region_score_map, score_thresh=5)
    region_quads = list()
    for idx in np.unique(region_segmentation_map):
        if idx == 0:
            continue

        pred_region_mask = (region_segmentation_map == idx).astype("uint8") * 255
        mask = _get_largest_overlapping_mask(mask1=pred_region_mask, masks=poly_masks)
        # mask = _get_masked_image(img=pred_region_mask, mask=polygon_mask)
        if mask.sum() != 0:
            quad = _get_minimum_area_bounding_rotated_rectangle(mask)
            region_quads.append(quad)
    return region_quads


def get_pseudo_score_map(img, quads):
    # quads=region_quads
    gaussian_map = _get_2d_isotropic_gaussian_map()
    pseudo_score_map = _get_canvas_same_size_as_image(
        img=_convert_to_2d(img), black=True
    )
    for quad in quads:
        # quad=quads[-3]
        # if len(np.unique(quad[:, 0])) != 4 or len(np.unique(quad[:, 1])) != 4:
        #     continue

        score_map = generate_score_map(img=img, quad=quad, gaussian_map=gaussian_map)
        # show_image(score_map)

        pseudo_score_map = np.maximum(pseudo_score_map, score_map)
    # show_image(pseudo_score_map, img)
    return pseudo_score_map


if __name__ == "__main__":
    cuda = torch.cuda.is_available()

    interim = load_craft_checkpoint(cuda)

    label_path = "/Users/jongbeomkim/Downloads/train_labels.json"
    with open(label_path, mode="r") as f:
        labels = json.load(f)
        for trg in tqdm(list(labels.keys())):
            trg = "gt_1342"
            img_path = f"/Users/jongbeomkim/Downloads/train_images/{trg}.jpg"
            img = load_image(img_path)

            pred_region, pred_affinity = _infer(img=img, craft=interim, cuda=cuda)
            # show_image(pred_region, img)

            r = _get_canvas_same_size_as_image(img=_convert_to_2d(img), black=True)
            a = _get_canvas_same_size_as_image(img=_convert_to_2d(img), black=True)
            for word in labels[trg]:
                poly=np.array(word["points"])
                word = labels[trg][1]
                region_quads = _get_region_quadlilaterals(
                    img=img, region_score_map=pred_region, poly=np.array(word["points"])
                )
                pseudo_region = get_pseudo_score_map(img=img, quads=region_quads)
                show_image(pseudo_region, img)
                r = np.maximum(r, pseudo_region)

                # region_quads
                affinity_quads = get_affinity_quadlilaterals(region_quads)
                pseudo_affinity = get_pseudo_score_map(img=img, quads=affinity_quads)
                show_image(pseudo_affinity, img)
                a = np.maximum(a, pseudo_affinity)
            # show_image(r, img)

            save_image(img1=r, img2=img, path=f"/Users/jongbeomkim/Downloads/score_maps/{trg}_pseudo_region.png")
            save_image(img1=a, img2=img, path=f"/Users/jongbeomkim/Downloads/score_maps/{trg}_pseudo_afiinity.png")

