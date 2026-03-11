# Naive Watermarking Baseline for PAN 26

This directory contains a naive baseline for PAN 26 on text watermarking. This baseline appends a naive watermark `_xy123_` to each text and looks for this text during the detection.

## High-Level Overview

We use [TIRA.io](https://www.tira.io/task-overview/text-watermarking-panclef-2026) for submissions.

On a high level, code submissions to TIRA require that your git repository is self-contained and contains all information to build your software into a Docker image. You can use this directory as a minimal working example, and please do not hesitate to contact the PAN team either in Discord or on TIRA in case you have problems, we are happy to help you organize your code so that it is submittable to TIRA.

TIRA performs three steps for a code-submission (possible within Github Actions):

- Build the Docker image from the source code in the repository
- Verify that the software produces valid outputs on a tiny spot-check dataset
- When the software is valid, it is uploaded to TIRA (together with metadata to track which version of the code yielded which software)

Therefore, this directory contains a `Dockerfile` that specifies how the software is compiled and `baseline.py` file is the actual to-be-executed baseline.

## Running Everything Manually

The watermarking task runs in three steps, first, your approach adds a watermark to the text. Second, an obfuscation approach potentially modifies the text that was watermarked by your system. Finally, your system is asked to predict which of the texts are watermarked or not.

Step by step, this would look like:

1.) Run the watermarking
```
./baseline.py watermark ../spot-check-dataset/ 01-watermarked
```

2.) Obfuscate the watermark
```
../obfuscation-baseline/obfuscate.py 01-watermarked/ ../spot-check-dataset/ 02-obfuscated
```

3.) Detect if watermark is in the texts
```
./baseline.py detect 02-obfuscated/ 03-detection
```

The final result (e.g., via `head -3 03-detection/detected-text.jsonl`) contains lines like:
```
{"id":"9a28d103-7b2e-43da-b511-2efb5f91975f","truth_label":"not-watermarked","label":0.0}
{"id":"4fcae8a8-f5e6-4ccd-9dae-10b609a47cfc","truth_label":"not-watermarked","label":0.0}
{"id":"91a6fd5b-dd14-4725-83ef-0ef6309f9b0e","truth_label":"not-watermarked","label":0.0}
```

## Submission To TIRA

To submit this baseline to TIRA (please ensure your tira client is up-to date via `pip3 install --upgrade tira`), you can run this command (`--dry-run` ensures that we first only test locally):

```
tira-cli code-submission \
	--task text-watermarking-panclef-2026 \
	--dataset spot-check-dataset-20260311-training \
	--path . \
	--set 'watermark_command=/baseline.py watermark $inputDataset $outputDir' \
	--set 'detect_command=/baseline.py detect $inputDataset $outputDir' \
	--dry-run
```

If everything looks good, you can remove the `--dry-run` flag, a valid upload looks like:


