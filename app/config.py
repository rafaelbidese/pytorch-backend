DETECTION_TRESHOLD = 0.8

MODEL_WEIGHTS = "model_final.pth"
MODEL_FRAMEWORK = "detectron2"
MODEL_STR = "resnet50_fpn"

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