from segmentation import Detectron2MAL, PytorchMAL


class MAL:
    def __init__(self):
        self.model = get_model(
            framework="detectron2", model_str="resnet50_fpn", weights="model_final.pth"
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
