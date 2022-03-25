from PIL import Image
import torch
import torchvision
import torchvision.transforms as T
from utils import to_coco


def get_transforms(x):
    return T.Compose(
        [
            T.ToTensor(),
        ]
    )(x)


class MAL:
    def __init__(self):
        self.model = torchvision.models.detection.maskrcnn_resnet50_fpn(
            pretrained_backbone=False, pretrained=False
        )
        checkpoint = torch.load("maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth")
        self.model.load_state_dict(checkpoint, strict=False)

        self.model.eval()

    def predict(self, image):
        x = Image.open(image).convert("RGB")
        w, h = x.size
        x = get_transforms(x).unsqueeze(0)
        pred = self.model(x)[0]
        pred = {k: pred[k].tolist() for k in pred.keys()}
        return to_coco(pred, w, h)
