DETECTION_TRESHOLD = 0.8


# Custom model configuration
MODEL_FRAMEWORK = "detectron2"
NUM_CLASSES = 1
MODEL_WEIGHTS = "model_final.pth"

# PyTorch specific
PYTORCH_MODEL_STR = "resnet50_fpn"
# Detectron2 specific
DETECTRON2_CONFIG = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml"

# MASKRCNN_CLASSES = "BG,person,bicycle,car,motorcycle,airplane,\
#          bus,train,truck,boat,traffic light,\
#          fire hydrant,stop sign,parking meter,bench,bird,\
#          cat,dog,horse,sheep,cow,elephant,bear,\
#          zebra,giraffe,backpack,umbrella,handbag,tie,\
#          suitcase,frisbee,skis,snowboard,sports ball,\
#          kite,baseball bat,baseball glove,skateboard,\
#          surfboard,tennis racket,bottle,wine glass,cup,\
#          fork,knife,spoon,bowl,banana,apple,\
#          sandwich,orange,broccoli,carrot,hot dog,pizza,\
#          donut,cake,chair,couch,potted plant,bed,\
#          dining table,toilet,tv,laptop,mouse,remote,\
#          keyboard,cell phone,microwave,oven,toaster,\
#          sink,refrigerator,book,clock,vase,scissors,\
#          teddy bear,hair drier,toothbrush".split(
#     ","
# )

MASKRCNN_CLASSES = "root".split(",")
