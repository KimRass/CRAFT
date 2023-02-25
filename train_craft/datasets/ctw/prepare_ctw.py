from pathlib import Path
import cv2
from collections import defaultdict
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import jsonlines
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

from process_images import (
    load_image,
    save_image,
    _convert_to_2d,
    _get_canvas_same_size_as_image,
    _get_width_and_height,
    _dilate_mask,
    _get_image_cropped_by_bboxes,
    _reverse_jet_colormap,
    _downsample_image
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


def get_cropped_images_and_labels(img_path, jsonl_path, size=200, sigma=0.5, crop=False, px_thresh=100_000, dilate=2):
    img, line = parse_jsonl_file(img_path=img_path, jsonl_path=jsonl_path)

    gaussian_map = _get_2d_isotropic_gaussian_map(width=size, height=size, sigma=sigma)
    region_score_map = get_region_score_map(img=img, annots=line["annotations"], gaussian_map=gaussian_map)
    affinity_score_map = get_affinity_score_map(img=img, annots=line["annotations"], gaussian_map=gaussian_map)
    if not crop:
        return img, region_score_map, affinity_score_map
    else:
        _, region_mask = cv2.threshold(src=region_score_map, thresh=100, maxval=255, type=cv2.THRESH_BINARY)
        n_labels, segmentation_map, stats, _ = cv2.connectedComponentsWithStats(
            image=_convert_to_2d(region_mask), connectivity=4
        )
        dilated_region_mask = _get_canvas_same_size_as_image(img=region_mask, black=True)
        for k in range(1, n_labels):
            width = stats[k, cv2.CC_STAT_WIDTH]
            height = stats[k, cv2.CC_STAT_HEIGHT]
            smaller = min(width, height)

            dilated_label = _dilate_mask(mask=(segmentation_map == k), kernel_shape=(smaller, smaller), iterations=dilate)
            dilated_region_mask = np.maximum(dilated_region_mask, dilated_label)
        # show_image(dilated_region_mask, img)

        _, _, stats, _ = cv2.connectedComponentsWithStats(image=_convert_to_2d(dilated_region_mask), connectivity=4)
        data = defaultdict(list)
        for xmin, ymin, width, height, pixel_count in stats[1:]:
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


class CTWDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.ls_prefix = sorted(
            list({file.stem.split("_", 1)[0] for file in self.data_dir.glob("*")})
        )
        self.transform_img = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ]
        )
        self.transform_score_map = T.ToTensor()
    def __len__(self):
        return len(self.ls_prefix)

    def __getitem__(self, idx):
        prefix = self.ls_prefix[idx]
        img = load_image(self.data_dir/f"{prefix}_image.jpg")
        region_score_map = _downsample_image(
            _reverse_jet_colormap(
                load_image(self.data_dir/f"{prefix}_region.png")
            )
        )
        affinity_score_map = _downsample_image(
            _reverse_jet_colormap(
                load_image(self.data_dir/f"{prefix}_affinity.png")
            )
        )

        img = self.transform_img(img)
        region_score_map = self.transform_score_map(region_score_map)
        affinity_score_map = self.transform_score_map(affinity_score_map)
        return img, region_score_map, affinity_score_map


if "__name__" == "__main__":
    # out_dir = Path("D:/ctw_out")
    out_dir = Path("/Users/jongbeomkim/Downloads/out2")
    # jsonl_path = "D:/ctw-annotations/train.jsonl"
    jsonl_path = "/Users/jongbeomkim/Downloads/ctw-annotations/train.jsonl"

    # in_dir = Path("D:/ctw-trainval-01-of-26")
    in_dir = Path("/Users/jongbeomkim/Downloads/ctw-trainval-17-of-26")
    for img_path in tqdm(sorted(list(in_dir.glob("**/*.jpg")))):
        img, region_score_map, affinity_score_map = get_cropped_images_and_labels(
            img_path=img_path, jsonl_path=jsonl_path, sigma=0.25, crop=False
        )
        save_image(img1=img, path=out_dir/f"{img_path.stem}_image.jpg")
        save_image(img1=region_score_map, path=out_dir/f"{img_path.stem}_region.png")
        save_image(img1=affinity_score_map, path=out_dir/f"{img_path.stem}_affinity.png")

        # data = get_cropped_images_and_labels(
        #     img_path=img_path, jsonl_path=jsonl_path, sigma=0.25, crop=True, px_thresh=50_000, dilate=2
        # )
        # for idx, (img, region_score_map, affinity_score_map) in enumerate(
        #     zip(data["image"], data["region_score_map"], data["affinity_score_map"])
        # ):
        #     save_image(img1=img, path=out_dir/f"{img_path.stem}_{str(idx).zfill(2)}_image.jpg")
        #     save_image(img1=region_score_map, path=out_dir/f"{img_path.stem}_{str(idx).zfill(2)}_region.png")
        #     save_image(img1=affinity_score_map, path=out_dir/f"{img_path.stem}_{str(idx).zfill(2)}_affinity.png")
