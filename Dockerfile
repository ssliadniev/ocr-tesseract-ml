FROM python:3.8-slim-buster

WORKDIR /app

COPY requirements.txt /app
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . /app/

CMD [ "python3", "./script.py" ]
