tira-run \
	--input-dataset pan23-text-detoxification/english-tiny-20231112-training \
	--image webis/clef24-text-detoxification-backtranslation \
	--command '/backtranslation_baseline.py --input ${inputDataset}/input.jsonl --output ${outputDir}/references.jsonl'