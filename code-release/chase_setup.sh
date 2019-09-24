#!/usr/bin/env bash

pip install -r requirements.txt
apt-get install python-cffi -y
cd core && make && cd ..
apt-get install python-opencv -y
apt-get install wget -y
eval ./download_cnn_weights.sh