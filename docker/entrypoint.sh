#!/bin/bash

set -e

groupadd -g ${CONTAINER_GID} -o ${CONTAINER_UNAME}
useradd --no-log-init -m -u ${CONTAINER_UID} -g ${CONTAINER_GID} -o -s /bin/bash ${CONTAINER_UNAME}

echo -e "export PYTHONPATH=${CONTAINER_WORKDIR}
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}
export PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:\${PATH}" >> /home/${CONTAINER_UNAME}/.bashrc

cd /home/${CONTAINER_UNAME}

if (( $# < 1 ))
then
    su - ${CONTAINER_UNAME}
else
    echo "$@"
    echo ${CONTAINER_UNAME}
    su ${CONTAINER_UNAME} -c "$@"
fi