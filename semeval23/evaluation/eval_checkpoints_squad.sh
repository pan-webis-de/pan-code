#!/bin/bash -e

usage()
{
	echo "Usage: ./eval_checkpoints.sh OPTIONS
			-m | --meth	method that was used for tuning ( squad | cbs20 )
					mandatory
			-t | --typ	type of clickbait ( phrase | passage | both)
					mandatory if meth = cbs20"
	exit 2
}

TYP=unset
METH=unset

PARSED_ARGUMENTS=$(getopt -a -n eval_abs_with_thresholds.sh -o t:m: --long typ:meth: -- "$@")

eval set -- "$PARSED_ARGUMENTS"
while :
do
	case "$1" in
		-t | --typ)  TYP=$2;  shift 2;;
		-m | --meth) METH=$2; shift 2;;
		# -- means the end of the arguments; drop this, and break out of the while loop
	    	--) shift; break ;;
	esac
done

cd /root/ceph_data/src/torch_huggingface

# set checkpoints of trained models to iterate through
if [ $METH = squad ]
then
	echo "EVALUATING on ${METH}"

elif [ $METH = cbs20 ] && ( [ $TYP = phrase ] || [ $TYP = passage ] || [ $TYP = both ] )
then
	echo "EVALUATING on ${METH} ${TYP}"
else
	echo "Invalid argument values given"
	usage
fi


for i in bertlarge_all #mpnetbase robertabase albertbasev2 bertbase-cased bertbase-uncased funneltransformer_small electrabase_discriminator #robertalarge debertalarge bigbird_robertalarge
do
	for j in 6000 7000 8000 9000 10000 11000 12000 13000 14000 15000 16000
	do

		export EVAL_PATH=/root/ceph_data/data/huggingface/$i/squad/checkpoint-$j

		if [ $METH = squad ]
		then
			## for validation on squad dev (~10500 entries)
			## models tuned only on squad
			#########################################

			mkdir ${EVAL_PATH}/eval_squad
			python3 run_qa_original.py --per_device_eval_batch_size=8 --overwrite_output_dir --do_eval --doc_stride 128 --max_seq_length 384 --dataset_name squad --model_name_or_path ${EVAL_PATH} --output_dir ${EVAL_PATH}/eval_squad/ |& tee ${EVAL_PATH}/eval_squad/squad_eval_log$j.txt

		fi

		if [ $METH = cbs20 ]
		then

			if [ $TYP = both ]
			then
				## for validation on cbs20 test with label 'passage' or 'phrase'  ( 182 entries),
				## models tuned only on squad
				#########################################

				mkdir ${EVAL_PATH}/eval_cbs20_${TYP}
				python3 run_qa.py --per_device_eval_batch_size=8 --overwrite_output_dir --do_eval --doc_stride 128 --max_seq_length 384 --dataset_loading_script load_sqlike_dataset_both.py --model_name_or_path ${EVAL_PATH} --output_dir ${EVAL_PATH}/eval_cbs20_${TYP}/ |& tee ${EVAL_PATH}/eval_cbs20_${TYP}/cbs_eval_log$j.txt

			else
				## for validation on cbs20 test with label 'passage' (102 entries),
				##                                         'phrase'  ( 80 entries),
				## models tuned only on squad
				#########################################

				mkdir ${EVAL_PATH}/eval_cbstest_${TYP}
				python3 run_qa.py --per_device_eval_batch_size=8 --overwrite_output_dir --do_eval --doc_stride 128 --max_seq_length 384 --dataset_loading_script load_sqlike_dataset_${TYP}.py --model_name_or_path ${EVAL_PATH} --output_dir ${EVAL_PATH}/eval_cbstest_${TYP}/ |& tee ${EVAL_PATH}/eval_cbstest_${TYP}/cbs_eval_log$j.txt

				## for validation on cbs20 1ktrain with label 'passage' (500 entries),
				##                                            'phrase'  (400 entries),
				## models tuned only on squad
				#########################################

				#mkdir ${EVAL_PATH}/eval_cbs1ktrain_${TYP}
				#python3 run_qa.py --per_device_eval_batch_size=8 --overwrite_output_dir --do_eval --doc_stride 128 --max_seq_length 384 --dataset_loading_script load_sqlike_1ktrain_${TYP}_eval.py --model_name_or_path ${EVAL_PATH} --output_dir ${EVAL_PATH}/eval_cbs1ktrain_${TYP}/ |& tee ${EVAL_PATH}/eval_cbs1ktrain_${TYP}/cbs_eval_log$j.txt
			fi

		fi
	done

	# last checkpoint is not in a "checkpoint"-dir
	export EVAL_PATH=/root/ceph_data/data/huggingface/$i/squad

	if [ $METH = squad ]
	then
		## for validation on squad dev (~10500 entries)
		#########################################

		mkdir ${EVAL_PATH}/eval_squad
		python3 run_qa_original.py --per_device_eval_batch_size=8 --overwrite_output_dir --do_eval --doc_stride 128 --max_seq_length 384 --dataset_name squad --model_name_or_path ${EVAL_PATH} --output_dir ${EVAL_PATH}/eval_squad/ |& tee ${EVAL_PATH}/eval_squad/squad_eval_log_checklast.txt

	fi

	if [ $METH = cbs20 ]
	then
		if [ $TYP = both ]
		then
			## for validation on cbs20 test with label 'passage' or 'phrase'  ( 182 entries),
			## models tuned only on squad
			#########################################

			mkdir ${EVAL_PATH}/eval_cbs20_${TYP}
			python3 run_qa.py --per_device_eval_batch_size=8 --overwrite_output_dir --do_eval --doc_stride 128 --max_seq_length 384 --dataset_loading_script load_sqlike_dataset_both.py --model_name_or_path ${EVAL_PATH} --output_dir ${EVAL_PATH}/eval_cbs20_${TYP}/ |& tee ${EVAL_PATH}/eval_cbs20_${TYP}/cbs_eval_log$j.txt

		else
			## for validation on cbs20 test with label 'passage' (102 entries),
			##                                         'phrase'  ( 80 entries),
			## models tuned only on squad
			#########################################

			mkdir ${EVAL_PATH}/eval_cbstest_${TYP}
			python3 run_qa.py --per_device_eval_batch_size=8 --overwrite_output_dir --do_eval --doc_stride 128 --max_seq_length 384 --dataset_loading_script load_sqlike_dataset_${TYP}.py --model_name_or_path ${EVAL_PATH} --output_dir ${EVAL_PATH}/eval_cbstest_${TYP}/ |& tee ${EVAL_PATH}/eval_cbstest_${TYP}/cbs_eval_log_checklast.txt

			## for validation on cbs20 1ktrain with label 'passage' (500 entries),
			##                                            'phrase'  (400 entries),
			## models tuned only on squad
			#########################################

			#mkdir ${EVAL_PATH}/eval_cbs1ktrain_${TYP}
			#python3 run_qa.py --per_device_eval_batch_size=8 --overwrite_output_dir --do_eval --doc_stride 128 --max_seq_length 384 --dataset_loading_script load_sqlike_1ktrain_${TYP}_eval.py --model_name_or_path ${EVAL_PATH} --output_dir ${EVAL_PATH}/eval_cbs1ktrain_${TYP}/ |& tee ${EVAL_PATH}/eval_cbs1ktrain_${TYP}/cbs_eval_log_checklast.txt
		fi

	fi
done
