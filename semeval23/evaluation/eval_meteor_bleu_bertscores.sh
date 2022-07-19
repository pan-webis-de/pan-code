#!/bin/bash -e

usage()
{
	echo "Usage: ./eval_checkpoints.sh OPTIONS
			-m | --meth	method that was used for tuning ( squad | cbs20 | squad+cbs20 )
					mandatory

			-t | --typ	type of tune ( phrase | passage | both )
					mandatory

			-d | --data	data that was evaluated on ( cbs1ktrain | cbstest )
					mandatory if meth = squad"
	exit 2
}

TYP=unset
METH=unset
DATA=unset

PARSED_ARGUMENTS=$(getopt -a -n eval_abs_with_thresholds.sh -o t:d:m: --long typ:meth: -- "$@")

eval set -- "$PARSED_ARGUMENTS"
while :
do
	case "$1" in
		-t | --typ)  TYP=$2;  shift 2;;
		-m | --meth) METH=$2; shift 2;;
		-d | --data) DATA=$2; shift 2;;
		# -- means the end of the arguments; drop this, and break out of the while loop
	    	--) shift; break ;;
	esac
done

# set checkpoints of trained models to iterate through
if [ $METH = squad ]
then
	echo "EVALUATING on ${METH} ${DATA}"
	CHECKPOINTS=($(seq 6000 1000 16000))

elif ([ $METH = cbs20 ] || [ $METH = squad+cbs20 ]) && [ $TYP = phrase ]
then
	echo "EVALUATING on ${METH} ${TYP}"
	CHECKPOINTS=($(seq 100 100 700))

elif ([ $METH = cbs20 ] || [ $METH = squad+cbs20 ]) && [ $TYP = passage ]
then
	echo "EVALUATING on ${METH} ${TYP}"
	CHECKPOINTS=($(seq 100 100 1100))

elif ([ $METH = cbs20 ] || [ $METH = squad+cbs20 ]) && [ $TYP = both ]
then
	echo "EVALUATING on ${METH} ${TYP}"
	CHECKPOINTS=($(seq 100 100 1900))
else
	echo "Invalid argument values given"
	usage
fi

# set directory name of trained models
case "$METH" in
	cbs20)       DIRNAME=cbs20_${TYP}_only ;;
	squad+cbs20) DIRNAME=cbs20_${TYP} ;;
esac

## calculate METEOR1.5 BLEU4 and BERTScore(albert-xxlarge-v2)
#############
for i in mpnetbase robertabase albertbasev2 bertbase-cased bertbase-uncased funneltransformer_small electrabase_discriminator #robertalarge debertalarge bigbird_robertalarge
do
	for j in "${CHECKPOINTS[@]}"
	do
		if [ $METH = squad ]
		then
			export EVAL_PATH=/root/ceph_data/data/huggingface/$i/squad/checkpoint-$j/eval_${DATA}_${TYP}/

		elif [ $METH = cbs20 ] || [ $METH = squad+cbs20 ]
		then
			export EVAL_PATH=/root/ceph_data/data/huggingface/$i/${DIRNAME}/checkpoint-$j/eval/
		else
			echo "Invalid -m (--meth) value given"
			usage
		fi

		python meteor-metric.py --preds ${EVAL_PATH}/eval_nbest_predictions.json --truth ~/ceph_data/data/clickbait-spoiling-corpus-20/cbs20_1k_train.jsonl --mode huggingface --meteor_dir /root/ceph_data/src/meteor-1.5/ --output_dir ${EVAL_PATH}

		bert-score -r ${EVAL_PATH}/truths.txt -c ${EVAL_PATH}/preds.txt -m albert-xxlarge-v2 -s |& tee ${EVAL_PATH}/bertscore_scores.txt

		python multi_purpose.py --fun make_all_json --bleu_scores ${EVAL_PATH}/bleu4_scores.json --meteor_scores ${EVAL_PATH}/meteor_scores.txt --BERTScore_scores ${EVAL_PATH}/bertscore_scores.txt --output_dir ${EVAL_PATH}
	done

	# last checkpoint is not in a 'checkpoint'-dir
	if [ $METH = squad ]
	then
		export EVAL_PATH=/root/ceph_data/data/huggingface/$i/squad/eval_${DATA}_${TYP}/

	elif [ $METH = cbs20 ] || [ $METH = squad+cbs20 ]
	then
		export EVAL_PATH=/root/ceph_data/data/huggingface/$i/${DIRNAME}/eval/
	else
		echo "Invalid -m (--meth) value given"
		usage
	fi

	python meteor-metric.py --preds ${EVAL_PATH}/eval_nbest_predictions.json --truth ~/ceph_data/data/clickbait-spoiling-corpus-20/cbs20_1k_train.jsonl --mode huggingface --meteor_dir /root/ceph_data/src/meteor-1.5/ --output_dir ${EVAL_PATH}

	bert-score -r ${EVAL_PATH}/truths.txt -c ${EVAL_PATH}/preds.txt -m albert-xxlarge-v2 -s |& tee ${EVAL_PATH}/bertscore_scores.txt

	python multi_purpose.py --fun make_all_json --bleu_scores ${EVAL_PATH}/bleu4_scores.json --meteor_scores ${EVAL_PATH}/meteor_scores.txt --BERTScore_scores ${EVAL_PATH}/bertscore_scores.txt --output_dir ${EVAL_PATH}
done

## eval with thresholded BLEU-4, METEOR, BERTScore
###############
bash eval_abs_with_thresholds.sh -t $TYP -d $DATA -m $METH |& tee /root/ceph_data/data/huggingface/results_in_scores/thresholded_${METH}_ble_met_ber_${DATA}_${TYP}.txt

## make csv from scores
###############

python make_csv_from_eval_logs.py --fun bmb --eval_path /root/ceph_data/data/huggingface/results_in_scores/thresholded_${METH}_ble_met_ber_${DATA}_${TYP}.txt --output_path /root/ceph_data/data/huggingface/results_in_scores/ble_met_ber_${METH}_${DATA}_${TYP}.csv
