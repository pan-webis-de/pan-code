## Submit to TIRA

To submit this baseline to TIRA, follow the instructions in the [documentation](https://docs.tira.io/participants/participate.html#submitting-your-submission).
On a high level, you need to perform these steps:

1. Join the task https://www.tira.io/task-overview/pan25-generated-plagiarism-detection
2. Click "Submit" on the task page
3. Choose "Code Submission" and click "New Submission"
4. Follow the steps shown in the Web UI (Download the TIRA client, authenticate, make the submission)
5. Run the submission using the Web UI

### Code submission
To submit this baseline to TIRA, please run:
```
tira-cli code-submission --path . --task pan25-generated-plagiarism-detection --dataset llm-plagiarism-detection-spot-check-20250521-training --command '/pan12-text-alignment-baseline.py $inputDataset $outputDir'
```

#### Running a submission on TIRA
After executing the command above, you will be shown the name of your code submission. Under "Code Submissions" in the Web UI, 
you can select the submission based on the name (and potentially rename it using the "Edit" button). 

Now that your submission is available on TIRA, you can run it on different datasets by using the dropdown menus in the Web UI (see also the [documentation](https://docs.tira.io/participants/participate.html#execute-your-submission)).

