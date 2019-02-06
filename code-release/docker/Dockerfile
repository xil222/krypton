FROM nvidia/cuda:9.2-cudnn7-devel-ubuntu16.04
LABEL maintainer "Supun Nakandala <snakanda@eng.ucsd.edu>"

RUN apt-get update && apt-get install python-cffi curl python-dev -y
RUN curl -O https://bootstrap.pypa.io/get-pip.py && python get-pip.py && rm get-pip.py

ADD requirements.txt /
RUN pip install -r requirements.txt