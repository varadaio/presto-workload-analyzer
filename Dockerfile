FROM python:3.7

COPY ./requirements.txt /app/requirements.txt
WORKDIR /app

RUN  pip3 install --no-cache-dir -r requirements.txt
COPY . /app
