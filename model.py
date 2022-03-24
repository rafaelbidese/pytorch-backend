from PIL import Image
import torch
import torchvision
import torchvision.transforms as T
import json


def to_coco(pred):
    # write a function to get the masks and find the countours
    # https://discuss.pytorch.org/t/pytorch-image-segmentation-mask-polygons/87054
    # contours, hierarchy = cv2.findContours(gray_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    return pred["boxes"]


def get_transforms(x):
    return T.Compose(
        [
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
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
        x = get_transforms(x).unsqueeze(0)
        pred = self.model(x)[0]
        pred = {k: pred[k].tolist() for k in pred.keys()}

        return to_coco(pred)
