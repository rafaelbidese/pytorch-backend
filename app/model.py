from segmentation import Detectron2MAL, PytorchMAL
from config.config import MODEL_FRAMEWORK

class MAL:
    def __init__(self):
        self.model = get_model(
            framework=MODEL_FRAMEWORK.lower(),
        )

    def predict(self, image):
        return self.model.predict(image)

def get_model(framework):
    model = None
    if framework == "pytorch":
        model = PytorchMAL()

    if framework == "detectron2":
        model = Detectron2MAL()

    return model
