# Pytorch-backend for COCO-Annotator

This backend service to perform predictions for model assisted labelling used with the [COCO-Annotator](https://github.com/jsbroks/coco-annotator) interface.


# Installation instructions

Clone this repository:

`git clone https://github.com/rafaelbidese/pytorch-backend.git`

Navigate to the repository folder:

`cd pytorch-backend`

And run the docker compose file (if you don't have the images created, it will create them by default):

`docker compose up`

# Setup COCO-Annotator

On your COCO-Annotator, log in, create a category, a dataset and load your images into the dataset. Start annotating an image to open the editor.

On the left upright menu bar, click on the Image Settings (cog icon) and configure the Annotate API.

By default, this API will be accessible at:

`http://localhost:9001/predict`

Copy and paste the address in the COCO-Annotator Annotate API field.

You can check if the server is running by accessing:

`http://localhost:9001/`


# Configuring your custom model

Off the box, we support two frameworks as backends: Detectron2 and Pytorch. 

- Open `config/config.py` and configure the model `MODEL_FRAMEWORK`, `NUM_CLASSES` and each framework specific variables. 
- Place your pre-trained model weights at the folder `models`.
- Update your model weight file name at `MODEL_WEIGHTS`.

Update `MASKRCNN_CLASSES` to match the categories that you want to predict from COCO-Annotator and your trained model.

# COCO-Annotator

COCO Annotator is a web-based image annotation tool designed for versatility and ease of use for efficiently label images to create training data for image localization and object detection.

## Setting up COCO-Annotator

To setup your COCO-Annotator please follow the instructions [here](https://github.com/jsbroks/coco-annotator/wiki/Getting-Started)

## How to use COCO-Annotator

Follow the instructions [here](https://github.com/jsbroks/coco-annotator/wiki/Usage) to learn how to use COCO-Annotator.
