from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine.defaults import DefaultPredictor
from PIL import Image
import numpy as np
import torchvision
import torch
from config.config import DETECTION_TRESHOLD, DETECTRON2_CONFIG, PYTORCH_MODEL_STR, NUM_CLASSES
import torchvision.transforms as T

from utils import mask_to_area, mask_to_segmentation, to_json, get_weights

class PytorchMAL:
    def __init__(self):
        if PYTORCH_MODEL_STR.lower() == "resnet50_fpn":
            self.model = torchvision.models.detection.maskrcnn_resnet50_fpn(
                pretrained_backbone=False, pretrained=False, num_classes=NUM_CLASSES
            )
            checkpoint = torch.load(get_weights())
            self.model.load_state_dict(checkpoint, strict=False)
            self.model.eval()

    def predict(self, image):
        x = Image.open(image).convert("RGB")
        w, h = x.size
        x = self._get_transforms(x).unsqueeze(0)
        with torch.no_grad():
            pred = self.model(x)[0]
        pred = {k: pred[k].tolist() for k in pred.keys()}
        return self._to_coco(pred, w, h)

    def _to_coco(self, pred, width, height):
        # ['boxes', 'labels', 'scores', 'masks']
        boxes, labels, masks, areas = [], [], [], []
        mask_polygons = [mask_to_segmentation(mask) for mask in pred["masks"]]
        mask_areas = [mask_to_area(mask) for mask in pred["masks"]]
        for b, l, s, p, a in zip(
            pred["boxes"], pred["labels"], pred["scores"], mask_polygons, mask_areas
        ):
            if s > DETECTION_TRESHOLD:
                boxes += [b]
                labels += [l]
                masks += [p]
                areas += [a]

        return to_json(boxes, labels, masks, areas, width, height)

    def _get_transforms(x):
        return T.Compose(
            [
                T.ToTensor(),
            ]
        )(x)


class Detectron2MAL:
    def __init__(self):
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(DETECTRON2_CONFIG))
        # cfg.MODEL.DEVICE = "cpu"
        cfg.MODEL.DEVICE = "cuda"
        cfg.MODEL.WEIGHTS = get_weights()
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = NUM_CLASSES
        self.model = DefaultPredictor(cfg)

    def predict(self, image):
        x = Image.open(image).convert("RGB")
        h, w = x.size
        x = np.uint8(np.array(x))
        pred = self.model(x)
        return self._to_coco(pred, w, h)

    def _to_coco(self, pred, width, height):
        instances = pred["instances"]
        confident_detections = instances[instances.scores > DETECTION_TRESHOLD]
        boxes = [box.tolist() for box in confident_detections.pred_boxes]
        labels = [int(x) for x in confident_detections.pred_classes.numpy()]
        masks = [
            mask_to_segmentation(mask.unsqueeze(0).numpy())
            for mask in confident_detections.pred_masks
        ]
        areas = [
            mask_to_area(mask.unsqueeze(0).numpy())
            for mask in confident_detections.pred_masks
        ]

        return to_json(boxes, labels, masks, areas, width, height)
