# Evaluation of Approaches for the shared tasks at PAN@CLEF 2023

## Development

- Run the unit tests via `make tests`.
- Build the docker image via `make build-docker-image`
- Publish the docker image via `make publish-docker-image`

## Integration in TIRA

Add this to TIRA via the image `kaaage/pan23-evaluator:0.0.1`:
- the command `bash -c 'python3 /clef23/evaluation/trigger-detection/evaluator.py --input_run $inputRun/labels.jsonl --truth $inputDataset/labels.jsonl --output_prototext ${outputDir}'`.
- the command (change for subtasks 1-3) `bash -c 'python3 /clef23/evaluation/profiling-cryptocrurrency-influencers/evaluation_script_subtask1.py -s $inputRun -g $inputDataset -o ${outputDir}'`.
