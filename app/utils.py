import os
import cv2
import numpy as np
import random
from config.config import MASKRCNN_CLASSES, MODEL_WEIGHTS
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
        annotation.bbox = box
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


class dotdict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def get_random_color():
    return ["#" + "".join([random.choice("ABCDEF0123456789") for _ in range(6)])][
        0
    ].lower()


def get_weights():
    weight_folder = "./models"
    weight_file = MODEL_WEIGHTS
    path_to_weight = os.path.join(weight_folder, weight_file)
    return path_to_weight