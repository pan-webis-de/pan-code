IMAGE_VERSION=0.0.11

build-docker-image:
	docker build -t webis/pan-clickbait-spoiling-evaluator:${IMAGE_VERSION} .

run-evaluation-jupyter-lab:
	docker run --rm -ti -p 8888:8888 --entrypoint jupyter-lab webis/pan-clickbait-spoiling-evaluator:${IMAGE_VERSION} --ip 0.0.0.0 --allow-root

publish-docker-image:
	docker push webis/pan-clickbait-spoiling-evaluator:${IMAGE_VERSION}

tests:
	nosetests

