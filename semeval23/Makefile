IMAGE_VERSION=0.0.10

build-docker-image:
	docker build -t webis/pan-clickbait-spoiling-evaluator:${IMAGE_VERSION} .

publish-docker-image:
	docker push webis/pan-clickbait-spoiling-evaluator:${IMAGE_VERSION}

tests:
	nosetests

