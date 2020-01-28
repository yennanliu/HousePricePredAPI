#!/bin/sh

docker build . -t house_pred_env
docker run -d -p 8000:8000 -it house_pred_env 
container_id="` docker ps -a | awk 'FNR == 2 {print $1}'`" && echo container_id = $container_id && image_id="` docker ps -a | awk 'FNR == 2 {print $2}'`" && echo image_id = $image_id
echo 'COMMIT & DEPLOY  : ' house_pred_env  && docker commit $container_id yennanliu/house_pred_env:V5 && docker push yennanliu/house_pred_env:V5