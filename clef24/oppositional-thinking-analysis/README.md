# Code for PAN24 Oppositional thinking analysis: Conspiracy theories vs critical thinking narratives

Run the evaluator locally (check requirements):

    ~$ python3 evaluator/evaluator.py {task1,task2} --gold <path_to_file_with_labels> --predictions <path_to_predictions_file> --outdir <path_to_folder_for_saving_results>

Run in Docker container:

    # create a Docker image DOCKER_IMAGE using the Dockerfile    

    ~$ docker run --rm \
        -v /path/to/gold.json:/gold.json \
        -v /path/to/predictions.json:/predictions.json \
        -v /path/to/output:/out \
        DOCKER_IMAGE \
        evaluator {task1,task2} \ 
        --gold /gold.json --predictions /predictions.json --outdir /out
