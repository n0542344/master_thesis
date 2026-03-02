#!/bin/bash


# Build the image
docker build -t gridsearch_img_0207 .

# Run the container with a volume mount
docker run -d \
  --cpus=25 \
  --memory=100g \
  --name gridsearch_ctn_0207 \
  -v $(pwd)/results:/model/results \
  -v $(pwd)/logs:/model/logs \
  gridsearch_img_0207 