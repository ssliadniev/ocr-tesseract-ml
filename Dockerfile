FROM python:3.8-slim-buster

WORKDIR /app

COPY requirements.txt /app
RUN pip install --upgrade pip && pip install -r requirements.txt
RUN apt update && apt install -y libsm6 libxext6
RUN apt-get -y install tesseract-ocr-swe

COPY . /app
