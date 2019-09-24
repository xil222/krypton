FROM nvidia/cuda:9.2-cudnn7-devel-ubuntu16.04
LABEL maintainer "Supun Nakandala <snakanda@eng.ucsd.edu>"

RUN apt-get update && apt-get install python-cffi curl python-dev -y
RUN curl -O https://bootstrap.pypa.io/get-pip.py && python get-pip.py && rm get-pip.py

ADD requirements.txt /

RUN pip install -r requirements.txt
RUN apt-get update
RUN apt-get install git -y
RUN git clone https://github.com/scnakandala/krypton.git
RUN apt-get install python-cffi -y
RUN cd krypton/code-release/core && make
RUN apt-get install python-opencv -y
RUN apt-get install wget -y
RUN cd krypton/code-release && ./download_cnn_weights.sh