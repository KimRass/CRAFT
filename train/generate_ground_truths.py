import numpy as np
import cv2
import sympy.geometry as gm
from sympy import Point2D

from process_images import (
    _convert_to_2d,
    _get_canvas_same_size_as_image,
    _get_width_and_height
)


def _get_2d_isotropic_gaussian_map(width, height, sigma=0.5):
    x, y = np.meshgrid(
  np.linspace(-1, 1, width), np.linspace(-1, 1, height)
    )
    d = np.sqrt(x ** 2 + y ** 2)
    mu = 0
    gaussian_map = np.exp(
  -(d - mu) ** 2 / (2 * sigma ** 2)
    )

    gaussian_map *= 255
    gaussian_map = gaussian_map.astype("uint8")
    return gaussian_map


def get_region_score_map(img, line, gaussian_map):
    width, height = _get_width_and_height(img)
    pts1 = np.float32([[0, 0], [size, 0], [size, size], [0, size]])

    region_score_map = _get_canvas_same_size_as_image(_convert_to_2d(img), black=True)
    for word in line["annotations"]:
        for char in word:
            pts2 = np.float32(char["polygon"])
            M = cv2.getPerspectiveTransform(src=pts1, dst=pts2)
            output = cv2.warpPerspective(src=gaussian_map, M=M, dsize=(width, height))

            region_score_map = np.maximum(region_score_map, output)
            return region_score_map


def _get_intersection_of_quarliateral(p11, p21, p12, p22):
    line1 = gm.Line(
        Point2D(list(map(int, p11))), Point2D(list(map(int, p12)))
    )
    line2 = gm.Line(
        Point2D(list(map(int, p21))), Point2D(list(map(int, p22)))
    )
    inter = line1.intersection(line2)
    return np.array(inter[0].evalf())


def get_affinity_score_map(img, line, gaussian_map):
    width, height = _get_width_and_height(img)
    pts1 = np.float32([[0, 0], [size, 0], [size, size], [0, size]])

    affinity_score_map = _get_canvas_same_size_as_image(_convert_to_2d(img), black=True)
    for word in line["annotations"]:
        for idx, char in enumerate(word):
            p11, p21, p12, p22 = char["polygon"]
            centroid = _get_intersection_of_quarliateral(p11=p11, p21=p21, p12=p12, p22=p22)

            affinity_p21 = (np.array(p11) + np.array(p21) + centroid) / 3
            affinity_p12 = (np.array(p12) + np.array(p22) + centroid) / 3

            if idx != 0:
                pts2 = np.float32([affinity_p11, affinity_p21, affinity_p12, affinity_p22])
                M = cv2.getPerspectiveTransform(src=pts1, dst=pts2)
                output = cv2.warpPerspective(src=gaussian_map, M=M, dsize=(width, height))

                affinity_score_map = np.maximum(affinity_score_map, output)
            affinity_p11 = affinity_p21
            affinity_p22 = affinity_p12
    return affinity_score_map


if "__name__" == "__main__":
    size = 200
    gaussian_map = _get_2d_isotropic_gaussian_map(width=size, height=size)
    region_score_map = get_region_score_map(img=img, line=line, gaussian_map=gaussian_map)
    affinity_score_map = get_affinity_score_map(img=img, line=line, gaussian_map=gaussian_map)
    show_image(region_score_map, img)
    show_image(affinity_score_map, img)