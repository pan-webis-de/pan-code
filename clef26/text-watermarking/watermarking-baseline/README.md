tira-cli code-submission \
	--task text-watermarking-panclef-2026 \
	--dataset spot-check-dataset-20260311-training \
	--path . \
	--set 'watermark_command=/baseline.py watermark $inputDataset $outputDir' \
	--set 'detect_command=/baseline.py detect $inputDataset $outputDir' \
	--dry-run
