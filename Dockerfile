FROM python:3.10-slim

ENV FLASK_ENV=production

ENV FLASK_RUN_PORT=9001

COPY requirements.txt /

RUN pip3 install --upgrade pip
RUN pip3 install -r /requirements.txt

COPY . /app

WORKDIR /app

ENTRYPOINT ["./gunicorn_starter.sh"]