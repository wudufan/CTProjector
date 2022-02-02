#!/bin/bash

ENV_NAME=ct_projector

conda create -y -n ${ENV_NAME} -c anaconda -c simpleitk \
python=3.8 \
tensorflow-gpu=2.4.1 \
cupy=8.3.0 \
numpy \
scipy \
flake8 \
matplotlib \
simpleitk

conda install -y -n ${ENV_NAME} -c conda-forge jupyterlab