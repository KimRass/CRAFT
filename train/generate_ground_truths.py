import numpy as np
import cv2
import sympy.geometry as gm

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
    line1 = gm.Line(gm.Point(p11), gm.Point(p12))
    line2 = gm.Line(gm.Point(p21), gm.Point(p22))
    intersection = line1.intersection(line2)
    x, y = intersection[0].evalf()
    return int(x), int(y)


if "__name__" == "__main__":
    size = 200
    gaussian_map = _get_2d_isotropic_gaussian_map(width=size, height=size)
    get_region_score_map(img=img, line=line, gaussian_map=gaussian_map)
    show_image(region_score_map, img)
    show_image(region_score_map)
    show_image(img)

    for char in word:
        p11, p21, p12, p22 = char["polygon"]
        x, y = _get_intersection_of_quarliateral(*char["polygon"])
        p11, p21, (x, y)
        # img[y, x, :] = np.array([255, 0, 0])
    show_image(img)
        