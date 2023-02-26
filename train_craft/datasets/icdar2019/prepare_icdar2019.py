import json
import numpy as np
import cv2
import torch

from process_images import (
    _get_canvas_same_size_as_image,
    _get_masked_image
)
from watershed import (
    _perform_watershed
)
from weakly_supervied_learning import (
    get_confidence_score
)


label_path = "/Users/jongbeomkim/Downloads/train_labels.json"
with open(label_path, mode="r") as f:
    labels = json.load(f)
    for trg in labels.keys():
        trg = "gt_0"
        img_path = f"/Users/jongbeomkim/Downloads/train_images/{trg}.jpg"
        img = load_image(img_path)

        label = labels[trg]
        conf_score_map = _get_canvas_same_size_as_image(img=img, black=True)
        for word in label:
            gt_length = len(word["transcription"])
            if gt_length > 0:
                # word=label[0]
                word_mask = _get_canvas_same_size_as_image(img=img, black=True)
                points = np.array(word["points"], dtype="int64")
                cv2.fillPoly(
                    img=word_mask,
                    pts=[np.array(word["points"], dtype="int64")],
                    color=(255, 255, 255)
                )
                masked_img = _get_masked_image(img=img, mask=word_mask)
                pred_region, pred_affinity = _infer(img=masked_img, craft=interim, cuda=cuda)

                pred_region_watershed = _perform_watershed(pred_region)
                pred_length = len(np.unique(pred_region_watershed)) - 1
                conf_score = get_confidence_score(gt_length=gt_length, pred_length=pred_length)
                conf_score_map[word_mask == 255] = int(conf_score * 255)
                print(gt_length, pred_length)
                print(conf_score)
# show_image(word_mask)
show_image(conf_score_map)
show_image(pred_region_watershed, img)
# cv2.polylines(
#     img=img,
#     # `"int64"`
#     pts=[points],
#     isClosed=True,
#     color=(255, 0, 0),
#     thickness=1
# )
