import numpy as np
from scipy.sparse import coo_matrix
import cv2
from skimage.feature import peak_local_max
from skimage.morphology import local_maxima
from skimage.segmentation import watershed

from process_images import (
    _get_width_and_height,
    _convert_to_2d
)


def _perform_watershed(score_map, score_thresh=50):
    # score_map = pred_region.copy()
    trimmed_score_map = score_map.copy()
    trimmed_score_map[trimmed_score_map < 190] = 0

    markers = local_maxima(image=trimmed_score_map)
    _, markers = cv2.connectedComponents(image=markers.astype("int8"), connectivity=8)

    _, region_mask = cv2.threshold(src=score_map, thresh=score_thresh, maxval=255, type=cv2.THRESH_BINARY)
    watersheded = watershed(image=-score_map, markers=markers, mask=_convert_to_2d(region_mask))
    # show_image(watersheded, img)
    return watersheded
# temp = _perform_watershed(pred_region)
# show_image(temp)



# def _get_local_maxima_coordinates(region_score_map):
#     # region_score_map=pred_region
#     _, region_mask = cv2.threshold(src=region_score_map, thresh=100, maxval=255, type=cv2.THRESH_BINARY)
#     # show_image(region_mask, img)
#     _, region_segmentation_map = cv2.connectedComponents(image=region_mask, connectivity=4)
#     show_image(region_segmentation_map, img)
#     local_max = peak_local_max(
#         image=region_score_map, min_distance=5, labels=region_segmentation_map, num_peaks_per_label=1
#     )
#     local_max = local_max[:, :: -1]
#     return local_max


# def _get_local_maxima_array(region_score_map):
#     local_max_coor = _get_local_maxima_coordinates(region_score_map)

#     _, height = _get_width_and_height(local_max_coor)
#     vals = np.array([1] * height)
#     rows = local_max_coor[:, 1]
#     cols = local_max_coor[:, 0]
#     local_max = coo_matrix(
#         (vals, (rows, cols)), shape=region_score_map.shape
#     ).toarray().astype("bool")
#     return local_max


# def _perform_watershed(score_map, score_thresh=30):
#     score_map = pred_region
#     local_max_arr = _get_local_maxima_array(score_map)
#     _, markers = cv2.connectedComponents(image=local_max_arr.astype("uint8"), connectivity=4)
#     show_image(markers)
#     np.unique(markers)
#     _, region_mask = cv2.threshold(
#         src=score_map, thresh=score_thresh, maxval=255, type=cv2.THRESH_BINARY
#     )
#     show_image(local_max_arr, img, 0.1)
#     segmentation_map = watershed(image=-score_map, markers=markers, mask=_convert_to_2d(region_mask))
#     # show_image(segmentation_map, img)
#     return segmentation_map
