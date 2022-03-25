FROM python:3.10-slim

RUN apt-get update && apt-get install -y python3-opencv

RUN apt-get install ffmpeg libsm6 libxext6  -y

ENV FLASK_ENV=production

ENV FLASK_RUN_PORT=9001

RUN python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

COPY requirements.txt /

RUN pip3 install --upgrade pip
RUN pip3 install -r /requirements.txt

COPY . /app

WORKDIR /app

ENTRYPOINT ["./gunicorn_starter.sh"]