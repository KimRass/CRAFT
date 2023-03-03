import numpy as np
import math
import cv2
from copy import deepcopy
from sympy import Point2D, Line
from itertools import combinations
from itertools import product

from train_craft.process_images import (
    _convert_to_2d,
    _get_canvas_same_size_as_image,
    _get_width_and_height,
    _sort_quadlilaterals
)


def _get_2d_isotropic_gaussian_map(w=200, h=200, sigma=0.5):
    x, y = np.meshgrid(
        np.linspace(-1, 1, w), np.linspace(-1, 1, h)
    )
    d = np.sqrt(x ** 2 + y ** 2)
    mu = 0
    gaussian_map = np.exp(
        -(d - mu) ** 2 / (2 * sigma ** 2)
    )

    gaussian_map *= 255
    gaussian_map = gaussian_map.astype("uint8")
    return gaussian_map


def generate_score_map(img, quad, gaussian_map, margin=0.3):
    quad
    gw, gh = _get_width_and_height(gaussian_map)
    left = gw * margin
    top = gh * margin
    right = gw * (1 - margin)
    bottom = gh * (1 - margin)
    pts1 = np.array(
        [[left, top], [right, top], [right, bottom], [left, bottom]], dtype="float32"
    )

    img_w, img_h = _get_width_and_height(img)
    # region_score_map = _get_canvas_same_size_as_image(img=_convert_to_2d(img), black=True)
    # quad
    M = cv2.getPerspectiveTransform(src=pts1, dst=quad.astype("float32"))
    output = cv2.warpPerspective(src=gaussian_map, M=M, dsize=(img_w, img_h))
    return output

    # region_score_map = np.maximum(region_score_map, output)
    # return region_score_map


# def _get_intersection_of_quarliateral(p11, p21, p12, p22):
#     line1 = Line(
#         Point2D(list(map(int, p11))), Point2D(list(map(int, p12)))
#     )
#     line2 = Line(
#         Point2D(list(map(int, p21))), Point2D(list(map(int, p22)))
#     )
#     inter = line1.intersection(line2)
#     return np.array(inter[0].evalf())


def _get_intersection_of_quarliateral(quad):
    # quad=quad1
    p11, p21, p12, p22 = quad
    line1 = Line(
        Point2D(list(map(int, p11))), Point2D(list(map(int, p12)))
    )
    line2 = Line(
        Point2D(list(map(int, p21))), Point2D(list(map(int, p22)))
    )
    inter = line1.intersection(line2)
    inter = np.array(inter[0].evalf(), dtype="int64")
    return inter


def generate_affinity_score_map(img, annots, gaussian_map, margin=0.3):
    gw, gh = _get_width_and_height(gaussian_map)
    left = gw * margin
    top = gh * margin
    right = gw * (1 - margin)
    bottom = gh * (1 - margin)
    pts1 = np.array(
        [[left, top], [right, top], [right, bottom], [left, bottom]], dtype="float32"
    )
    # pts1 = np.array([[0, 0], [gw, 0], [gw, gh], [0, gh]], dtype="float32")

    img_w, img_h = _get_width_and_height(img)
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
                output = cv2.warpPerspective(src=gaussian_map, M=M, dsize=(img_w, img_h))

                affinity_score_map = np.maximum(affinity_score_map, output)
            cp11 = cp21
            cp22 = cp12
    return affinity_score_map


def get_affinity_quadlilateral(quad1, quad2):
    # quad1=quads[-1]
    # quad2=quads[-2]
    # quad1
    inter1 = _get_intersection_of_quarliateral(quad1)
    inter2 = _get_intersection_of_quarliateral(quad2)

    # writing_dir = "horizontal" if abs(inter1[0] - inter2[0]) >= abs(inter1[1] - inter2[1]) else "vertical"
    # if writing_dir == "horizontal":
    #     if inter1[0] > inter2[0]:
    #         inter1, inter2 = inter2, inter1
    #         quad1, quad2 = quad2, quad1

    p1 = (quad1[0, :] + quad1[1, :] + inter1) / 3
    p2 = (quad2[0, :] + quad2[1, :] + inter2) / 3
    p3 = (inter2 + quad2[2, :] + quad2[3, :]) / 3
    p4 = (inter1 + quad1[2, :] + quad1[3, :]) / 3
    affinity_quad = np.stack((p1, p2, p3, p4)).astype("int64")
    return affinity_quad



def get_affinity_quadlilaterals(quads):
    # quads = region_quads
    if quads:
        quads = _sort_quadlilaterals(quads)

        affinity_quads = list()
        for quad1, quad2 in zip(quads, quads[1:]):
            try:
                cv2.polylines(
                    img=img,
                    pts=[get_affinity_quadlilateral(quad1, quad2)],
                    isClosed=True,
                    color=(255, 0, 0),
                    thickness=1
                )
                show_image(img)
                # affinity_quads.append(
                #     get_affinity_quadlilateral(quad1, quad2)
                # )
            except Exception:
                continue
    else:
        affinity_quads = list()
    return affinity_quads


