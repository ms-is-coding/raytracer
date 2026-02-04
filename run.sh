#!/usr/bin/env bash

IMG_NAME="raytracer-dev"

docker build -t $IMG_NAME .

mkdir -p .dev_home

docker run --rm -it \
    --name raytracer-container \
    --device /dev/dri:/dev/dri \
    -v "$PWD":/raytracer \
    -v "$PWD/.dev_home":/home/dev \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    --user 1000:1000 \
    $IMG_NAME

