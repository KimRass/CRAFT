import numpy as np
import cv2


def _get_2d_isotropic_gaussian_map(width, height):
    x, y = np.meshgrid(np.linspace(-1, 1, width), np.linspace(-1, 1, height))
    d = np.sqrt(x * x + y * y)
    mu = 0
    sigma = 0.5
    gaussian_map = np.exp(-( (d-mu)**2 / ( 2.0 * sigma**2 ) ) )

    gaussian_map *= 255
    gaussian_map = gaussian_map.astype("uint8")
    return gaussian_map




width = 100
height = 200
gaussian_map = _get_2d_isotropic_gaussian_map(width=width, height=height)

pts1 = np.float32(list(each_char(anno))[1]['polygon'])
pts2 = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
pts2
mtrx = cv2.getPerspectiveTransform(src=pts1, dst=pts2)
result = cv2.warpPerspective(img, mtrx, (width, height))
show_image(result)

pts1
pts2