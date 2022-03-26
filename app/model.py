import os
from segmentation import Detectron2MAL, PytorchMAL
from config import MODEL_WEIGHTS, MODEL_FRAMEWORK, MODEL_STR


class MAL:
    def __init__(self):
        weight_folder = "./models"
        weight_file = MODEL_WEIGHTS
        path_to_weight = os.path.join(weight_folder, weight_file)
        self.model = get_model(
            framework=MODEL_FRAMEWORK.lower(), model_str=MODEL_STR.lower(), weights=path_to_weight
        )

    def predict(self, image):
        return self.model.predict(image)


def get_model(
    framework="pytorch",
    model_str="resnet50_fpn",
    num_classes=2,
    weights="maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth",
):
    model = None
    if framework == "pytorch":
        model = PytorchMAL(model_str, num_classes, weights)

    if framework == "detectron2":
        model = Detectron2MAL(model_str, num_classes, weights)

    return model
