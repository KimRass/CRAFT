import numpy as np
import cv2
from PIL import Image
from itertools import product
from pathlib import Path
import requests


def _convert_to_2d(img):
    if img.ndim == 3:
        return img[:, :, 0]
    else:
        return img


def _convert_to_3d(img):
    if img.ndim == 2:
        return np.dstack([img, img, img])
    else:
        return img


def _get_width_and_height(img):
    if img.ndim == 2:
        height, width = img.shape
    else:
        height, width, _ = img.shape
    return width, height


def _convert_to_pil(img):
    if not isinstance(img, Image.Image):
        img = Image.fromarray(img)
    return img


def _convert_to_array(img):
    img = np.array(img)
    return img


def _apply_jet_colormap(img):
    img_jet = cv2.applyColorMap(src=(255 - img), colormap=cv2.COLORMAP_JET)
    return img_jet


def _reverse_jet_colormap(img):
    gray_values = np.arange(256, dtype=np.uint8)
    color_values = list(map(tuple, _apply_jet_colormap(gray_values).reshape(256, 3)))
    color_to_gray_map = dict(zip(color_values, gray_values))

    out = np.apply_along_axis(lambda bgr: color_to_gray_map[tuple(bgr)], axis=2, arr=img)
    return out


def _repaint_segmentation_map(segmentation_map, n_color_values=3):
    canvas_r = _get_canvas_same_size_as_image(segmentation_map, black=True)
    canvas_g = _get_canvas_same_size_as_image(segmentation_map, black=True)
    canvas_b = _get_canvas_same_size_as_image(segmentation_map, black=True)

    color_vals = list(range(50, 255 + 1, 255 // n_color_values))
    perm = list(product(color_vals, color_vals, color_vals))[1:]
    perm = perm[:: 2] + perm[1:: 2]

    remainder_map = segmentation_map % len(perm) + 1
    for remainder, (r, g, b) in enumerate(perm, start=1):
        canvas_r[remainder_map == remainder] = r
        canvas_g[remainder_map == remainder] = g
        canvas_b[remainder_map == remainder] = b
    canvas_r[segmentation_map == 0] = 0
    canvas_g[segmentation_map == 0] = 0
    canvas_b[segmentation_map == 0] = 0

    dstacked = np.dstack([canvas_r, canvas_g, canvas_b])
    return dstacked


def _preprocess_image(img):
    if img.dtype == "int32":
        img = _repaint_segmentation_map(img)

    if img.dtype == "bool":
        img = img.astype("uint8") * 255
        
    if img.ndim == 2:
        if (
            np.array_equal(np.unique(img), np.array([0, 255])) or
            np.array_equal(np.unique(img), np.array([0])) or
            np.array_equal(np.unique(img), np.array([255]))
        ):
            img = _convert_to_3d(img)
        else:
            img = _apply_jet_colormap(img)
    return img


def _blend_two_images(img1, img2, alpha=0.5):
    img1 = _convert_to_pil(img1)
    img2 = _convert_to_pil(img2)
    img_blended = Image.blend(im1=img1, im2=img2, alpha=alpha)
    return _convert_to_array(img_blended)


def _get_canvas_same_size_as_image(img, black=False):
    if black:
        return np.zeros_like(img).astype("uint8")
    else:
        return (np.ones_like(img) * 255).astype("uint8")


def _dilate_mask(mask, kernel_shape=(3, 3), iterations=1):
    kernel = cv2.getStructuringElement(
        shape=cv2.MORPH_RECT, ksize=(kernel_shape[1], kernel_shape[0])
    )
    if mask.dtype == "bool":
        mask = mask.astype("uint8") * 255
    mask = cv2.dilate(src=mask, kernel=kernel, iterations=iterations)
    return mask


def _get_image_cropped_by_bboxes(img, xmin, ymin, xmax, ymax):
    if img.ndim == 3:
        return img[ymin: ymax, xmin: xmax, :]
    else:
        return img[ymin: ymax, xmin: xmax]


def _resize_image(img, width, height):
    resized_img = cv2.resize(src=img, dsize=(width, height))
    return resized_img


def _downsample_image(img):
    return cv2.pyrDown(img)


def load_image(url_or_path="", gray=False):
    url_or_path = str(url_or_path)

    if "http" in url_or_path:
        img_arr = np.asarray(
            bytearray(requests.get(url_or_path).content), dtype="uint8"
        )
        if not gray:
            img = cv2.imdecode(img_arr, flags=cv2.IMREAD_COLOR)
            img = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2RGB)
        else:
            img = cv2.imdecode(img_arr, flags=cv2.IMREAD_GRAYSCALE)
    else:
        if not gray:
            img = cv2.imread(url_or_path, flags=cv2.IMREAD_COLOR)
            img = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2RGB)
        else:
            img = cv2.imread(url_or_path, flags=cv2.IMREAD_GRAYSCALE)
    return img


def save_image(img1, img2=None, alpha=0.5, path="") -> None:
    copied_img1 = _preprocess_image(
        _convert_to_array(img1.copy())
    )
    if img2 is None:
        img_arr = copied_img1
    else:
        copied_img2 = _convert_to_array(
            _preprocess_image(
                _convert_to_array(img2.copy())
            )
        )
        img_arr = _convert_to_array(
            _blend_two_images(img1=copied_img1, img2=copied_img2, alpha=alpha)
        )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if img_arr.ndim == 3:
        cv2.imwrite(
            filename=str(path), img=img_arr[:, :, :: -1], params=[cv2.IMWRITE_JPEG_QUALITY, 100]
        )
    elif img_arr.ndim == 2:
        cv2.imwrite(
            filename=str(path), img=img_arr, params=[cv2.IMWRITE_JPEG_QUALITY, 100]
        )


def show_image(img1, img2=None, alpha=0.5):
    img1 = _convert_to_pil(
        _preprocess_image(
            _convert_to_array(img1)
        )
    )
    if img2 is None:
        img1.show()
    else:
        img2 = _convert_to_pil(
            _preprocess_image(
                _convert_to_array(img2)
            )
        )
        img_blended = Image.blend(im1=img1, im2=img2, alpha=alpha)
        img_blended.show()


def _invert_image(mask):
    return cv2.bitwise_not(mask)


def _get_masked_image(img, mask, invert=False):
    img = _convert_to_array(img)
    mask = _convert_to_2d(
        _convert_to_array(mask)
    )
    if invert:
        mask = _invert_image(mask)
    return cv2.bitwise_and(src1=img, src2=img, mask=mask.astype("uint8"))


def _get_minimum_area_bounding_rotated_rectangle(mask):
    contours, _ = cv2.findContours(image=mask, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
    rect = cv2.minAreaRect(contours[0])
    rect = cv2.boxPoints(rect)
    return rect.astype("int64")


def convert_to_polygon_to_mask(img, poly):
    poly_mask = _get_canvas_same_size_as_image(_convert_to_2d(img), black=True)
    cv2.fillPoly(
        img=poly_mask,
        pts=[poly],
        color=(255, 255, 255),
    )
    return poly_mask