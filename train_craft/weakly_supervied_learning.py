import json
import numpy as np
import cv2
from pprint import pprint
from tqdm.auto import tqdm

from train_craft.process_images import (
    _get_canvas_same_size_as_image,
    _convert_to_2d,
    _get_masked_image,
    _get_minimum_area_bounding_rotated_rectangle,
    convert_to_polygon_to_mask
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


def _sort_points_in_quadliateral(pts):
	# initialzie a list of coordinates that will be ordered
	# such that the first entry in the list is the top-left,
	# the second entry is the top-right, the third is the
	# bottom-right, and the fourth is the bottom-left
	rect = np.zeros((4, 2), dtype = "float32")
	# the top-left point will have the smallest sum, whereas
	# the bottom-right point will have the largest sum
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
	# now, compute the difference between the points, the
	# top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
	# return the ordered coordinates
	return rect


def _get_largest_overlapping_mask(mask1, masks2):
    maxim = 0
    largest_mask = _get_canvas_same_size_as_image(mask1, black=True)
    for mask2 in masks2:
        overlap_mask = _get_masked_image(img=mask1, mask=mask2)
        summ = overlap_mask.sum()
        if summ > maxim:
            maxim = summ
            largest_mask = overlap_mask
    return largest_mask


def _get_region_quadlilaterals(img, region_score_map, polygon):
    # region_score_map = pred_region
    # polygon=np.array(word["points"])
    region_segmentation_map = _perform_watershed(region_score_map, score_thresh=5)
    polygon_mask = convert_to_polygon_to_mask(img=img, poly=np.array(polygon))
    # show_image(region_segmentation_map, img)
    # show_image(polygon_mask, img)

    # temp = _get_canvas_same_size_as_image(img=_convert_to_2d(img), black=True)
    region_quads = list()
    for idx in np.unique(region_segmentation_map):
        if idx == 0:
            continue

        pred_region_mask = (region_segmentation_map == idx).astype("uint8") * 255
        # show_image(pred_region_mask, img)
        # mask = _get_largest_overlapping_mask(mask1=pred_region_mask, masks2=[polygon_mask])
        mask = _get_masked_image(img=pred_region_mask, mask=polygon_mask)
        # show_image(mask, img)
        if mask.sum() != 0:
            # temp = np.maximum(temp, mask)
            quad = _get_minimum_area_bounding_rotated_rectangle(mask)
            quad = _sort_points_in_quadliateral(quad)
            region_quads.append(quad)
    # show_image(temp, img)
    return region_quads


def get_pseudo_score_map(img, quads):
    # quads=region_quads
    gaussian_map = _get_2d_isotropic_gaussian_map()
    pseudo_score_map = _get_canvas_same_size_as_image(
        img=_convert_to_2d(img), black=True
    )
    for quad in quads:
        # quad=quads[-3]
        if len(np.unique(quad[:, 0])) != 4 or len(np.unique(quad[:, 1])) != 4:
            continue

        score_map = generate_score_map(img=img, quad=quad, gaussian_map=gaussian_map)
        # show_image(score_map)

        pseudo_score_map = np.maximum(pseudo_score_map, score_map)
    return pseudo_score_map
        

# def straighten_curved_text(img, points):
#     # for word in label:
#     #     gt_length = len(word["transcription"])
#     #     if gt_length > 0:
#     n_points = len(points)

#     prev_w = 0
#     prev_h = 0
#     max_h = 0
#     canvas = np.zeros(shape=(500, 300, 3), dtype="uint8")
#     for idx in range(n_points // 2 - 1):
#         pts11 = points[idx]
#         pts12 = points[idx + 1]
#         pts13 = points[n_points - 2 - idx]
#         pts14 = points[n_points - 1 - idx]

#         pts1 = np.stack([pts11, pts12, pts13, pts14])
#         # w = int((np.linalg.norm(pts11 - pts12) + np.linalg.norm(pts14 - pts13)) / 2)
#         # h = int((np.linalg.norm(pts12 - pts13) + np.linalg.norm(pts14 - pts11)) / 2)
#         w = max(np.linalg.norm(pts11 - pts12), np.linalg.norm(pts14 - pts13))
#         h = max(np.linalg.norm(pts12 - pts13), np.linalg.norm(pts14 - pts11))

#         # pts2 = np.array([[0, 77 - h], [w, 77 - h], [w, 77], [0, 77]], dtype="float32")
#         pts2 = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype="float32")
#         M = cv2.getPerspectiveTransform(src=pts1, dst=pts2)

#         output = cv2.warpPerspective(src=img, M=M, dsize=(int(w), int(h)))
#         # output = cv2.warpPerspective(src=img, M=M, dsize=(w, 77))
#         # show_image(output)
#         canvas[0: h, prev_w: prev_w + w, :] = output
#         # canvas[prev_h // 2 - h // 2: prev_h // 2 - h // 2 + h, prev_w: prev_w + w, :] = output

#         prev_w += w
#         prev_h += h
#     show_image(canvas)
#     return canvas
# label = labels[trg]
# word = label[2]
# points = np.array(word["points"], dtype="float32")

# temp = straighten_curved_text(img, points)
# show_image(temp)


if __name__ == "__main__":
    label_path = "/Users/jongbeomkim/Downloads/train_labels.json"
    with open(label_path, mode="r") as f:
        labels = json.load(f)
        for trg in tqdm(labels.keys()):
            trg = "gt_4387"
            img_path = f"/Users/jongbeomkim/Downloads/train_images/{trg}.jpg"
            img = load_image(img_path)

            pred_region, pred_affinity = _infer(img=img, craft=interim, cuda=cuda)

            try:
                r = _get_canvas_same_size_as_image(img=_convert_to_2d(img), black=True)
                a = _get_canvas_same_size_as_image(img=_convert_to_2d(img), black=True)
                for word in labels[trg]:
                    region_quads = _get_region_quadlilaterals(img=img, region_score_map=pred_region, polygon=np.array(word["points"]))
                    pseudo_region = get_pseudo_score_map(img=img, quads=region_quads)
                    show_image(pseudo_region, img)
                    r = np.maximum(r, pseudo_region)
                    show_image(r, img)

                    affinity_quads = get_affinity_quadlilaterals(region_quads)
                    # affinity_quads = _sort_points_in_quadliateral(affinity_quads)
                    pseudo_affinity = get_pseudo_score_map(img=img, quads=affinity_quads)
                    a = np.maximum(a, pseudo_affinity)

                save_image(img1=r, img2=img, path=f"/Users/jongbeomkim/Downloads/score_maps/{trg}_pseudo_region.png")
                save_image(img1=a, img2=img, path=f"/Users/jongbeomkim/Downloads/score_maps/{trg}_pseudo_afiinity.png")
            except Exception:
                continue


            show_image(img)
            show_image(pseudo_region, img)
            show_image(pred_region, img)
            region_quads

quad
cv2.polylines(
    img=img,
    # pts=[q.astype("int64") for q in region_quads],
    pts=[quad.astype("int64")],
    isClosed=True,
    color=(255, 0, 0),
    thickness=1
)
show_image(img)