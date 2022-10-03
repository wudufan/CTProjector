#/bin/bash

export HOST_USER_ID="$(id -u)"
export HOST_GROUP_ID="$(id -g)"
export HOST_USER_NAME=${USER}
export CONTAINER_WORKDIR="/workspace/ct_projector"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SOURCE_DIR="$(dirname "$SCRIPT_DIR")"

docker compose run compile "/env/compile.sh"