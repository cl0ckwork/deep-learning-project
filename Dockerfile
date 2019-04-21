FROM continuumio/miniconda3

RUN mkdir /app
RUN mkdir /app/raw_data

WORKDIR /app

ADD requirements.loader.txt requirements.loader.txt
RUN pip install -r requirements.loader.txt

COPY . .

ENTRYPOINT python pre_process_main.py