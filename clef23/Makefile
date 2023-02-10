DOCKERHUB_USER=kaaage
IMAGE_VERSION=0.0.1
EVALUATOR_NAME=pan23-evaluator

build-docker-image:
	docker build -t ${DOCKERHUB_USER}/${EVALUATOR_NAME}:${IMAGE_VERSION} .

publish-docker-image:
	docker push ${DOCKERHUB_USER}/${EVALUATOR_NAME}:${IMAGE_VERSION}

run-evaluation-jupyter-lab:
	docker run --rm -ti -p 8888:8888 --entrypoint jupyter-lab ${DOCKERHUB_USER}/pan23-evaluator:${IMAGE_VERSION} --ip 0.0.0.0 --allow-root
