import numpy as np
import cv2
from PIL import Image
from itertools import product
from pathlib import Path
import requests
import math
from copy import deepcopy


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


def _get_image_cropped_by_rectangle(img, xmin, ymin, xmax, ymax):
    if img.ndim == 3:
        return img[ymin: ymax, xmin: xmax, :]
    else:
        return img[ymin: ymax, xmin: xmax]


def _resize_image(img, width, height):
    resized_img = cv2.resize(src=img, dsize=(width, height))
    return resized_img


def _downsample_image(img):
    return cv2.pyrDown(img)


def _upsample_image(img):
    return cv2.pyrUp(img)


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
    # for cnt in contours:
    #     cv2.drawContours(warped_img, [cnt], 0, (255, 0, 0), 1)
    # show_image(warped_img)
    rrect = cv2.minAreaRect(contours[0])
    rrect = cv2.boxPoints(rrect)
    rrect = sort_points_in_quadlilateral(rrect)
    rrect = rrect.astype("int64")
    return rrect


def convert_polygon_to_mask(img, poly):
    poly_mask = _get_canvas_same_size_as_image(_convert_to_2d(img), black=True)
    cv2.fillPoly(
        img=poly_mask,
        pts=[poly],
        color=(255, 255, 255),
    )
    return poly_mask


def draw_polygons(img, polys):
    copied_img = img.copy()
    for poly in polys:
        poly = poly.astype("int64")
        cv2.polylines(
            img=copied_img,
            pts=[poly],
            isClosed=True,
            color=(255, 0, 0),
            thickness=1
        )
        for point in poly:
            cv2.circle(img=copied_img, center=point, radius=2, color=(0, 255, 0), thickness=-1)
    return copied_img


def distance(p1, p2):
    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5


def _sort_points(points):
    min_tot_dist = math.inf
    for idx, cur_point in enumerate(deepcopy(points)):
        copied_points = deepcopy(points)
        visited = [idx]
        copied_points.remove(cur_point)
        tot_dist = 0
        # Find the nearest unvisited point and add it to the visited list
        while len(copied_points) > 0:
            nearest_point, dist = sorted(
                [(p, distance(cur_point, p)) for p in copied_points],
                key=lambda x: x[1]
            )[0]
            tot_dist += dist
            visited.append(points.index(nearest_point))
            copied_points.remove(nearest_point)
            cur_point = nearest_point
        
        if tot_dist < min_tot_dist:
            min_tot_dist = tot_dist
            selected_visited = visited
    return tuple(selected_visited)


def _sort_quadlilaterals(quads):
    centroids = [list(quad.mean(axis=0)) for quad in quads]
    order = _sort_points(centroids)
    return [quads[idx] for idx in order]


def sort_points_in_quadlilateral(quad):
    cx, cy = quad.mean(0)
    x, y = quad.T
    angles = np.arctan2(x - cx, y - cy) * 180 / math.pi + 90
    angles = np.where(angles >= 0, angles, 360 + angles)
    indices = np.argsort(-angles)
    return quad[indices]


def _make_segmentation_map_rectangle(segmentation_map):
    copied_segmentation_map = segmentation_map.copy()
    for label in range(1, np.max(copied_segmentation_map) + 1):
        segmentation_map_sub = (copied_segmentation_map == label)
        nonzero_x = np.where((segmentation_map_sub != 0).any(axis=0))[0]
        nonzero_y = np.where((segmentation_map_sub != 0).any(axis=1))[0]
        if nonzero_x.size != 0 and nonzero_y.size != 0:
            copied_segmentation_map[
                nonzero_y[0]: nonzero_y[-1] + 1, nonzero_x[0]: nonzero_x[-1] + 1
            ] = label
    return copied_segmentation_map


def _make_mask_rectangle(mask):
    copied_mask = mask.copy()

    nonzero_x = np.where((copied_mask != 0).any(axis=0))[0]
    nonzero_y = np.where((copied_mask != 0).any(axis=1))[0]
    if nonzero_x.size != 0 and nonzero_y.size != 0:
        copied_mask[
            nonzero_y[0]: nonzero_y[-1] + 1, nonzero_x[0]: nonzero_x[-1] + 1
        ] = 255
    return copied_mask


def _get_minimum_area_bounding_rectangle(mask):
    contours, _ = cv2.findContours(image=mask, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv2.boundingRect(contours[0])
    return x, y, x + w, y + h


def perform_perspective_transform(src_quad, dst_quad, src_img, out_resolution, cut=False):
    M = cv2.getPerspectiveTransform(src=src_quad, dst=dst_quad)
    if not cut:
        warped_img = cv2.warpPerspective(src=src_img, M=M, dsize=out_resolution)
    else:
        mask = _get_canvas_same_size_as_image(img=src_img, black=True)
        cv2.fillPoly(
            img=mask,
            pts=[src_quad.astype("int32")],
            color=(255, 255, 255),
        )
        masked_src_img = _get_masked_image(img=src_img, mask=mask)

        warped_img = cv2.warpPerspective(src=masked_src_img, M=M, dsize=out_resolution)
    return warped_img


# def straighten_curved_text(img, poly):
#     words = labels[trg]

#     img_w, img_h = _get_width_and_height(img)
#     gaussian_map = _get_2d_isotropic_gaussian_map()
#     xmin, ymin, xmax, ymax = _get_gaussian_map_core_rectangle(gaussian_map=gaussian_map, margin=margin)
#     core_rect = np.array(
#         [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]], dtype="float32"
#     )

#     pseudo_region = _get_canvas_same_size_as_image(img=_convert_to_2d(img), black=True)
#     for word in words:
#         word=words[0]
#         poly = np.array(word["points"])
#         n_points = len(poly)

#         ls = list()
#         dst_quad = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype="float32")
#         for idx in range(n_points // 2 - 1):
#             # dr = draw_polygons(img, [poly])
#             # show_image(dr)
#             # idx=0
#             point1 = poly[idx]
#             point2 = poly[idx + 1]
#             point3 = poly[n_points - 2 - idx]
#             point4 = poly[n_points - 1 - idx]

#             subword_quad = np.array([point1, point2, point3, point4]).astype("float32")
#             # dst_w = int(max(np.linalg.norm(point1 - point2), np.linalg.norm(point4 - point3)))
#             # dst_h = int(max(np.linalg.norm(point2 - point3), np.linalg.norm(point4 - point1)))
#             # dst_quad = np.array([[0, 0], [dst_w, 0], [dst_w, dst_h], [0, dst_h]], dtype="float32")
#             # dst_quad = np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]], dtype="float32")

#             ls.append(subword_quad)
#             # xmin += 100
#             # xmax += 100

#         ls_warped_pred_region = list()
#         for subword_quad in ls:
#             # warped_subword = perform_perspective_transform(
#             #     src_quad=subword_quad, dst_quad=dst_quad, src_img=img, out_resolution=(dst_w, dst_h)
#             # )
#             # dr = draw_polygons(img, [subword_quad])
#             # show_image(dr)
#             warped_pred_region = perform_perspective_transform(
#                 src_quad=subword_quad, dst_quad=dst_quad, src_img=pred_region, out_resolution=(100, 100)
#             )
#             # show_image(warped_pred_region)
#             ls_warped_pred_region.append(warped_pred_region)
#         warped_pred_region = np.concatenate(ls_warped_pred_region, axis=1)
#         show_image(warped_pred_region)

#         # show_image(warped_pred_region)
#             # ls_warped_pred_region.append(warped_subword)
#         # dr = draw_polygons(img, polys)
#         # show_image(dr)
#         # show_image(np.concatenate(ls_warped_pred_region, axis=1))
#         # show_image(pred_region, img)
#             # warped_pred_affinity = perform_perspective_transform(
#             #     src_quad=subword_quad, dst_quad=dst_quad, src_img=pred_affinity, out_resolution=(dst_w, dst_h)
#             # )

#         watersheded = _perform_watershed(score_map=warped_pred_region, score_thresh=150)
#         # show_image(watersheded)
#             # show_image(watersheded)
#         canvas2 = _get_canvas_same_size_as_image(pred_region, black=True)
#         for label in np.unique(watersheded):
#             if label == 0:
#                 continue
#             pred_region_mask = (watersheded == label).astype("uint8") * 255
#             # show_image(pred_region_mask)
#             xmin, ymin, xmax, ymax = _get_minimum_area_bounding_rectangle(pred_region_mask)
#             canvas = np.zeros_like(watersheded, dtype="uint8")
#             canvas[ymin: ymax, xmin: xmax] = label
#             show_image(canvas.astype("int32"))

#             for idx, subword_quad in enumerate(ls):
#                 quad = np.array([[idx * 100, 0], [(idx + 1) * 100, 0], [(idx + 1) * 100, 100], [idx * 100, 100]], dtype="float32")
#                 # quad
#                 temp = perform_perspective_transform(
#                     src_quad=quad, dst_quad=subword_quad, src_img=canvas, out_resolution=(img_w, img_h), cut=True
#                 )
#                 # show_image(temp.astype("int32"))
#                 canvas2 = np.maximum(canvas2, temp)
#         show_image(canvas2.astype("int32"))
#             # show_image(canvas)


#             bounding_rect = np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]], dtype="float32")
#             bounding_rect

#             M1 = cv2.getPerspectiveTransform(src=core_rect, dst=bounding_rect)
#             M2 = cv2.getPerspectiveTransform(src=dst_quad, dst=subword_quad)
#             pseudo_subregion = cv2.warpPerspective(src=gaussian_map, M=np.matmul(M2, M1), dsize=(img_w, img_h))
#             pseudo_region = np.maximum(pseudo_region, pseudo_subregion)
#     show_image(pseudo_region, img)
    # save_image(img1=pseudo_region, img2=img, path="D:/pseudo_region.jpg")



    

# points = np.array(word["points"], dtype="float32")

# temp = straighten_curved_text(img, points)
# show_image(temp)


# if __name__ == "__main__":
#     polys = [
#         np.array(word["points"], dtype="int64") for word in label
#     ]
#     # dr = draw_polygons(img=img, polys=polys)
    
#     dr = draw_polygons(img=dr, polys=affinity_quads)
#     show_image(dr)