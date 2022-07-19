#!/bin/bash -e
		
export EVAL_PATH=/workspace/cbs20_dataset/all_scores.json

python3 /src/multi_purpose.py --fun calc_with_thresh --threshs phrase --meteor_scores ${EVAL_PATH} --train /workspace/cbs20_dataset/train_test/200_test.jsonl
