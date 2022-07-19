#!/bin/bash -e

usage()
{
	echo "Usage: ./eval_checkpoints_withcbstune.sh OPTIONS
			-t | --typ	type of clickbait ( phrase | passage | phrase_only | passage_only | both )"
	exit 2
}

TYP=unset
DATA=unset
METH=unset

PARSED_ARGUMENTS=$(getopt -a -n eval_abs_with_thresholds.sh -o t: --long typ: -- "$@")

eval set -- "$PARSED_ARGUMENTS"
while :
do
	case "$1" in
		-t | --typ)  TYP=$2;  shift 2;;
		# -- means the end of the arguments; drop this, and break out of the while loop
	    	--) shift; break ;;
	esac
done

cd /root/ceph_data/src/

# set checkpoints of trained models to iterate through
if [ $TYP = phrase ] || [ $TYP = phrase_only ]
then
	CHECKPOINTS=($(seq 200 200 2200)) #CHECKPOINTS=($(seq 100 100 700))

elif [ $TYP = passage ] || [ $TYP = passage_only ]
then
	CHECKPOINTS=($(seq 200 200 2200))

elif [ $TYP = both ]
then
	CHECKPOINTS=($(seq 100 100 1900))

else
	echo "Unsupported --typ value given"
	usage
fi


for i in robertalarge_all #mpnetbase robertabase albertbasev2 bertbase-cased bertbase-uncased funneltransformer_small electrabase_discriminator robertalarge debertalarge bigbird_robertalarge
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

		python3 run_qa.py --per_device_eval_batch_size=8 --overwrite_output_dir --do_eval --doc_stride 128 --max_seq_length 384 --dataset_loading_script load_sqlike_dataset_${TYP}.py --model_name_or_path /root/ceph_data/data/huggingface/$i/cbs20_${TYP}/checkpoint-$j/ --output_dir /root/ceph_data/data/huggingface/$i/cbs20_${TYP}/checkpoint-$j/eval/ |& tee /root/ceph_data/data/huggingface/$i/cbs20_${TYP}/checkpoint-$j/cbs_eval_log$j.txt

	done

	# last checkpoint is not in a "checkpoint"-dir
	python3 run_qa.py --per_device_eval_batch_size=8 --overwrite_output_dir --do_eval --doc_stride 128 --max_seq_length 384 --dataset_loading_script load_sqlike_dataset_${TYP}.py --model_name_or_path /root/ceph_data/data/huggingface/$i/cbs20_${TYP}/ --output_dir /root/ceph_data/data/huggingface/$i/cbs20_${TYP}/eval/ |& tee /root/ceph_data/data/huggingface/$i/cbs20_${TYP}/cbs_eval_log_checklast.txt

done
