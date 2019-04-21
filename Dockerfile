FROM continuumio/anaconda3:2019.03-alpine
MAINTAINER lukebogacz@gmail.com

RUN mkdir /app
WORKDIR /app
COPY . .

RUN pip install -r requirements.loader.txt
RUN pip freeze

ENTRYPOINT python pre_process_main.py --dry_run --check