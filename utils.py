import cv2
import numpy as np
import random
from config import MASKRCNN_CLASSES, DETECTION_TRESHOLD
from templates import ANNOTATIONS_TEMPLATE, CATEGORY_TEMPLATE, IMAGES_TEMPLATE


def to_json(boxes, labels, masks, areas, width, height):
    images = []
    image = dotdict(IMAGES_TEMPLATE)
    image.width = width
    image.height = height

    images += [image]

    categories = []
    for label in set(labels):
        category = dotdict(CATEGORY_TEMPLATE)
        category.id = label
        category.name = MASKRCNN_CLASSES[label]
        category.color = get_random_color()
        categories += [category]

    annotations = []
    for i, (box, label, mask, area) in enumerate(zip(boxes, labels, masks, areas)):
        annotation = dotdict(ANNOTATIONS_TEMPLATE)
        annotation.id = i
        annotation.category_id = label
        annotation.width = width
        annotation.height = height
        annotation.area = area
        annotation.segmentation = [mask]
        annotation.box = box
        annotation.color = get_random_color()
        annotations += [annotation]

    coco = {"annotations": annotations, "categories": categories, "images": images}
    return coco


def mask_to_segmentation(mask):
    contours = mask_to_contour(mask)
    contours_flat_list = []
    for contour in contours:
        contours_flat_list += np.array(contour).flatten().tolist()

    return contours_flat_list


def mask_to_area(mask):
    contours = mask_to_contour(mask)
    contours_area = []
    for contour in contours:
        contours_area += [cv2.contourArea(contour)]

    return contours_area


def mask_to_contour(mask):
    _, thresh = cv2.threshold(np.float32(mask).squeeze(0), 0.01, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(
        np.uint8(thresh), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    return contours


def to_coco(pred, width, height):
    # ['boxes', 'labels', 'scores', 'masks']
    boxes, labels, scores, masks, areas = [], [], [], [], []
    mask_polygons = [mask_to_segmentation(mask) for mask in pred["masks"]]
    mask_areas = [mask_to_area(mask) for mask in pred["masks"]]
    for b, l, s, p, a in zip(
        pred["boxes"], pred["labels"], pred["scores"], mask_polygons, mask_areas
    ):
        if s > DETECTION_TRESHOLD:
            boxes += [b]
            labels += [l]
            scores += [s]
            masks += [p]
            areas += [a]

    return to_json(boxes, labels, masks, areas, width, height)


class dotdict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def get_random_color():
    return ["#" + "".join([random.choice("ABCDEF0123456789") for _ in range(6)])][
        0
    ].lower()
