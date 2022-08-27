IMAGE_VERSION=0.0.2

build-docker-image:
	docker build -t webis/tira-ir-measures-evaluator:${IMAGE_VERSION} .

publish-docker-image:
	docker push webis/tira-ir-measures-evaluator:${IMAGE_VERSION}

tests:
	nosetests

