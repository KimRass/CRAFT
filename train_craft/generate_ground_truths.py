import numpy as np
import cv2
from sympy import Point2D, Line

from train_craft.process_images import (
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


def get_region_score_map(img, annots, gaussian_map, margin=0.3):
    gwidth, gheight = _get_width_and_height(gaussian_map)
    left = gwidth * margin
    top = gheight * margin
    right = gwidth * (1 - margin)
    bottom = gheight * (1 - margin)
    pts1 = np.array(
        [[left, top], [right, top], [right, bottom], [left, bottom]], dtype="float32"
    )

    img_width, img_height = _get_width_and_height(img)
    region_score_map = _get_canvas_same_size_as_image(img=_convert_to_2d(img), black=True)
    for word in annots:
        for char in word:
            pts2 = np.array(char["polygon"], dtype="float32")
            M = cv2.getPerspectiveTransform(src=pts1, dst=pts2)
            output = cv2.warpPerspective(src=gaussian_map, M=M, dsize=(img_width, img_height))

            region_score_map = np.maximum(region_score_map, output)
    return region_score_map


def _get_intersection_of_quarliateral(p11, p21, p12, p22):
    line1 = Line(
        Point2D(list(map(int, p11))), Point2D(list(map(int, p12)))
    )
    line2 = Line(
        Point2D(list(map(int, p21))), Point2D(list(map(int, p22)))
    )
    inter = line1.intersection(line2)
    return np.array(inter[0].evalf())


def get_affinity_score_map(img, annots, gaussian_map, margin=0.3):
    gwidth, gheight = _get_width_and_height(gaussian_map)
    left = gwidth * margin
    top = gheight * margin
    right = gwidth * (1 - margin)
    bottom = gheight * (1 - margin)
    pts1 = np.array(
        [[left, top], [right, top], [right, bottom], [left, bottom]], dtype="float32"
    )
    pts1 = np.array([[0, 0], [gwidth, 0], [gwidth, gheight], [0, gheight]], dtype="float32")

    img_width, img_height = _get_width_and_height(img)
    affinity_score_map = _get_canvas_same_size_as_image(img=_convert_to_2d(img), black=True)
    for word in annots:
        for idx, char in enumerate(word):
            p11, p21, p12, p22 = char["polygon"]
            inter = _get_intersection_of_quarliateral(p11=p11, p21=p21, p12=p12, p22=p22)

            cp21 = (np.array(p11) + np.array(p21) + inter) / 3
            cp12 = (np.array(p12) + np.array(p22) + inter) / 3

            if idx != 0:
                pts2 = np.array([cp11, cp21, cp12, cp22], dtype="float32")
                M = cv2.getPerspectiveTransform(src=pts1, dst=pts2)
                output = cv2.warpPerspective(src=gaussian_map, M=M, dsize=(img_width, img_height))

                affinity_score_map = np.maximum(affinity_score_map, output)
            cp11 = cp21
            cp22 = cp12
    return affinity_score_map
