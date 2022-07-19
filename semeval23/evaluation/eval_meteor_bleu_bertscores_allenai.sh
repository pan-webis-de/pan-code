#!/bin/bash -e

export EVAL_PATH=/workspace/cbs20_dataset/

python3 /src/meteor-metric.py --preds ${EVAL_PATH}/predictions/allenai_phrase_predictions.jsonl --truth ${EVAL_PATH}/cbs20_1k_train.jsonl --mode allenai --meteor_dir /workspace/meteor-1.5/ --output_dir ${EVAL_PATH}

bert-score -r ${EVAL_PATH}/truths.txt -c ${EVAL_PATH}/preds.txt -m albert-xxlarge-v2 -s |& tee ${EVAL_PATH}/bertscore_scores.txt

python3 /src/multi_purpose.py --fun make_all_json --bleu_scores ${EVAL_PATH}/bleu4_scores.json --meteor_scores ${EVAL_PATH}/meteor_scores.txt --BERTScore_scores ${EVAL_PATH}/bertscore_scores.txt --output_dir ${EVAL_PATH}

## eval with thresholded BLEU-4, METEOR, BERTScore
###############
bash /src/eval_abs_with_thresholds_allenai.sh |& tee ${EVAL_PATH}/thresholded_allenai_ble_met_ber.txt

## make csv from scores
###############

python3 /src/make_csv_from_eval_logs.py --fun bmb --eval_path ${EVAL_PATH}/thresholded_allenai_ble_met_ber.txt --output_path ${EVAL_PATH}/ble_met_ber_allenai.csv
