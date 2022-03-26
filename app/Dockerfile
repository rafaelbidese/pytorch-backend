FROM python:3.10-slim

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6 python3-opencv -y
RUN apt-get install git gcc g++ -y

ENV FLASK_ENV=production

ENV FLASK_RUN_PORT=9001

COPY requirements.txt /

RUN pip3 install --upgrade pip
RUN pip3 install -r /requirements.txt

RUN python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

COPY . /app

WORKDIR /app

ENTRYPOINT ["./gunicorn_starter.sh"]