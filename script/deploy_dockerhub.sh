#!/bin/sh

#################################################################
# SCRIPT DEPLOY DOCKER IMAGE TO DOCKER HUB 
#################################################################

echo ' ---------------- DEPLOY TO DOCKER HUB ----------------'

declare -a docker_images=(".")

echo "$REGISTRY_PASS" | docker login  --username $REGISTRY_USER --password $REGISTRY_PASS && echo "docker login OK" || echo "docker login failed"

for docker_image in "${docker_images[@]}"
	do 
		instance_name=house_pred_env
		container_id="` docker ps -a | awk 'FNR == 2 {print $1}'`" && echo container_id = $container_id && image_id="` docker ps -a | awk 'FNR == 2 {print $2}'`" && echo image_id = $image_id 
		# docker deploy 
		echo 'COMMIT & DEPLOY  : ' $docker_image  && docker commit $container_id yennanliu/$instance_name:V1 && docker push yennanliu/$instance_name:V1
	done 