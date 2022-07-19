#!/bin/bash -e

usage()
{
	echo "Usage: ./eval_abs_with_thresholds.sh OPTIONS
			-m | --meth	method that was used for tuning ( squad | cbs20 | squad+cbs20 )
					mandatory

			-t | --typ	type of clickbait ( phrase | passage | both )
					mandatory if meth= ( cbs20 | squad+cbs20 )

			-d | --data	data that was predicted on ( cbs1ktrain | cbstest )
					mandatory if meth= squad"
	exit 2
}

TYP=unset
DATA=unset
METH=unset

PARSED_ARGUMENTS=$(getopt -a -n eval_abs_with_thresholds.sh -o t:d:m: --long typ:data:meth: -- "$@")

eval set -- "$PARSED_ARGUMENTS"
while :
do
	case "$1" in
		-t | --typ)  TYP=$2;  shift 2;;
		-d | --data) DATA=$2; shift 2;;
		-m | --meth) METH=$2; shift 2;;
		# -- means the end of the arguments; drop this, and break out of the while loop
	    	--) shift; break ;;
	esac
done

#cd /root/ceph_data/src/

# set checkpoints of trained models to iterate through
if [ $METH = squad ]
then
	CHECKPOINTS=($(seq 6000 1000 16000))
fi

if [ $METH = cbs20 ] || [ $METH = squad+cbs20 ]
then
	if [ $TYP = passage ]
	then
		CHECKPOINTS=($(seq 100 100 1100))

	elif [ $TYP = phrase ]
	then
		CHECKPOINTS=($(seq 100 100 700))

	elif [ $TYP = both ]
	then
		CHECKPOINTS=($(seq 100 100 1900))
	fi
fi

# set directory name of trained models
case "$METH" in
	cbs20)       DIRNAME=cbs20_${TYP}_only ;;
	squad+cbs20) DIRNAME=cbs20_${TYP} ;;
esac


for i in mpnetbase robertabase #albertbasev2 bertbase-cased bertbase-uncased funneltransformer_small electrabase_discriminator #robertalarge debertalarge bigbird_robertalarge
do
	for j in "${CHECKPOINTS[@]}"
	do
	
		## for validation set of cbs20 1ktrain with label 'passage' (500 entries),
		##                             test    with label 'passage' (102 entries),
		##                             1ktrain with label 'phrase'  (400 entries),
		##                             test    with label 'phrase'  ( 80 entries),
		## models tuned only on squad
		##              only on cbs20
		##              on squad and cbs20
		#########################################
		if [ $METH = squad ]
		then
			export EVAL_PATH=/root/ceph_data/data/huggingface/$i/squad/checkpoint-$j/eval_${DATA}_${TYP}/all_scores.json
		else
			export EVAL_PATH=/root/ceph_data/data/huggingface/$i/${DIRNAME}/checkpoint-$j/eval/all_scores.json
		fi

		echo $i Checkpoint$j $METH $DATA $TYP
		python multi_purpose.py --fun calc_with_thresh --threshs ${TYP} --meteor_scores ${EVAL_PATH} --train /root/ceph_data/data/clickbait-spoiling-corpus-20/train_test/200_test.jsonl
	done

	# last checkpoint is not in a "checkpoint"-dir
	if [ $METH = squad ]
	then
		export EVAL_PATH=/root/ceph_data/data/huggingface/$i/squad/eval_${DATA}_${TYP}/all_scores.json
	else
		export EVAL_PATH=/root/ceph_data/data/huggingface/$i/${DIRNAME}/eval/all_scores.json
	fi

	echo $i Checkpointlast $METH $DATA $TYP
	python multi_purpose.py --fun calc_with_thresh --threshs ${TYP} --meteor_scores ${EVAL_PATH} --train /root/ceph_data/data/clickbait-spoiling-corpus-20/train_test/200_test.jsonl
done
