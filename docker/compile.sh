#!/bin/bash

export PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
cd ${CONTAINER_WORKDIR}/src/ct_projector/kernel

make clean
make all