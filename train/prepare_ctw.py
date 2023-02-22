from pathlib import Path
import cv2
from collections import defaultdict
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import jsonlines
from tqdm.auto import tqdm

from process_images import (
    load_image,
    save_image,
    _convert_to_2d,
    _get_canvas_same_size_as_image,
    _get_width_and_height,
    _dilate_mask,
    _get_image_cropped_by_bboxes
)
from generate_ground_truths import (
    _get_2d_isotropic_gaussian_map,
    get_region_score_map,
    get_affinity_score_map
)


def parse_jsonl_file(img_path, jsonl_path):
    img_path = Path(img_path)
    img = load_image(img_path)
    with jsonlines.open(jsonl_path) as f:
        for line in f.iter():
            if line["file_name"] == img_path.name:
                break
    return img, line


def get_cropped_images_and_labels(img_path, jsonl_path, size=200, sigma=0.5, px_thresh=100_000):
    img, line = parse_jsonl_file(img_path=img_path, jsonl_path=jsonl_path)

    gaussian_map = _get_2d_isotropic_gaussian_map(width=size, height=size, sigma=sigma)
    region_score_map = get_region_score_map(img=img, annots=line["annotations"], gaussian_map=gaussian_map)
    affinity_score_map = get_affinity_score_map(img=img, annots=line["annotations"], gaussian_map=gaussian_map)

    _, region_mask = cv2.threshold(src=region_score_map, thresh=100, maxval=255, type=cv2.THRESH_BINARY)
    n_labels, segmentation_map, stats, _ = cv2.connectedComponentsWithStats(
        image=_convert_to_2d(region_mask), connectivity=4
    )
    dilated_region_mask = _get_canvas_same_size_as_image(img=region_mask, black=True)
    for k in range(1, n_labels):
        width = stats[k, cv2.CC_STAT_WIDTH]
        height = stats[k, cv2.CC_STAT_HEIGHT]
        smaller = min(width, height)

        dilated_label = _dilate_mask(mask=(segmentation_map == k), kernel_shape=(smaller, smaller), iterations=6)
        dilated_region_mask = np.maximum(dilated_region_mask, dilated_label)

    _, _, stats, _ = cv2.connectedComponentsWithStats(image=_convert_to_2d(dilated_region_mask), connectivity=4)
    data = defaultdict(list)
    for xmin, ymin, width, height, pixel_count in stats[1:]:
        pixel_count
        if pixel_count >= px_thresh:
            img_patch = _get_image_cropped_by_bboxes(
                img=img, xmin=xmin, ymin=ymin, xmax=xmin + width, ymax=ymin + height
            )
            region_score_map_patch = _get_image_cropped_by_bboxes(
                img=region_score_map, xmin=xmin, ymin=ymin, xmax=xmin + width, ymax=ymin + height
            )
            affinity_score_map_patch = _get_image_cropped_by_bboxes(
                img=affinity_score_map, xmin=xmin, ymin=ymin, xmax=xmin + width, ymax=ymin + height
            )

            data["image"].append(img_patch)
            data["region_score_map"].append(region_score_map_patch)
            data["affinity_score_map"].append(affinity_score_map_patch)
    return data


if "__name__" == "__main__":
    out_dir = Path("/Users/jongbeomkim/Downloads/out2")
    jsonl_path="/Users/jongbeomkim/Downloads/ctw-annotations/train.jsonl"

    in_dir = Path("/Users/jongbeomkim/Downloads/ctw-trainval-17-of-26")
    for img_path in tqdm(list(in_dir.glob("**/*.jpg"))):
        data = get_cropped_images_and_labels(img_path=img_path, jsonl_path=jsonl_path, sigma=0.25)
        for idx, (img, region_score_map, affinity_score_map) in enumerate(
            zip(data["image"], data["region_score_map"], data["affinity_score_map"])
        ):
            save_image(img1=img, path=out_dir/f"{img_path.stem}_{str(idx).zfill(2)}_image.png")
            save_image(img1=region_score_map, path=out_dir/f"{img_path.stem}_{str(idx).zfill(2)}_region.png")
            save_image(img1=affinity_score_map, path=out_dir/f"{img_path.stem}_{str(idx).zfill(2)}_affinity.png")


    # img_path = "/Users/jongbeomkim/Downloads/ctw-trainval-17-of-26/3005039.jpg"
    # data = get_cropped_images_and_labels(img_path=img_path, jsonl_path=jsonl_path, sigma=0.25)
    # save_image(
    #     img1=data["region_score_map"][0], img2=data["image"][0], alpha=0.3, path="/Users/jongbeomkim/Downloads/temp1.jpg"
    # )
    # region, _, _ = get_score_maps(data["image"][0], craft, None, False)
    # save_image(
    #     img1=region, img2=data["image"][0], alpha=0.3, path="/Users/jongbeomkim/Downloads/temp2.jpg"
    # )