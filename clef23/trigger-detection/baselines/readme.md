# XGBoost Baseline for the shared task on Trigger Detection at PAN23

This baseline uses [Gradient Boosted Trees](https://xgboost.readthedocs.io) based on a tf-idf document vectors. 
The `baseline-xgboost-trainer.py` script trains and saves the model and can run a ablation study. 
The `baseline-xgboost-runner.py` script loads the saved model and makes predictions. 

A pre-build version of this image is uploaded at [mattiwiegmann/pan23-trigger-detection-baseline-xgboost:latest](https://hub.docker.com/repository/docker/mattiwiegmann/pan23-trigger-detection-baseline-xgboost/general). You can use this image to test the submission to TIRA from Step 3.

### Usage (Re-Run Prebuilt Baseline)

The official baseline (building instructions below) can be executed via the following command:

```
pip install tira
cd pan-code/clef23/trigger-detection
tira-run --image mattiwiegmann/pan23-trigger-detection-baseline-xgboost \
   --input-directory ${PWD}/sample-data/input \
   --command 'python3 /baseline/baseline-xgboost-runner.py --input-dataset-dir $inputDataset --output-dir $outputDir'
```

This should create an output in tira-output/labels.jsonl` like (`head -3 tira-output/labels.jsonl` gives):

```
{"work_id": "23796682", "labels": ["pornographic-content"]}
{"work_id": "23812675", "labels": ["pornographic-content"]}
{"work_id": "23800645", "labels": ["pornographic-content", "sexual-assault"]}
```

### Usage (Development)

1. Set up the training environment and train the model. Set the `-a` flag to run the ablation study instead.

    ```
    python3 -m venv venv && source venv/bin/activate
    pip install -r requirements.txt
    python3 baseline-xgboost-trainer.py \
        --training <path-to-training-dataset> \
        --validation <path-to-validation-dataset> \
        --savepoint ./models/xgb-baseline
    ```

2. Execute the runner script to make and save predictions.

    ```
    python3 baseline-xgboost-runner.py \
        --input-dataset-dir <path-to-test-dataset> \
        --output-dir <path-where-to-write-the-output> \
        --savepoint ./models/xgb-baseline
    ```

### Docker and Tira submission

For a software submission on tira, you can dockerize the trained model with the given dockerfile. 

1. After training the model as shown above, build the container to run the model:

    ```
   docker build -t <dockerhub-user>/pan23-trigger-detection-baseline-xgboost:latest -f dockerfile.xgb .
   ```

2. (optional) Upload the image to a public docker repository, i.e. dockerhub. 

   ```
   docker push <dockerhub-user>/pan23-trigger-detection-baseline-xgboost:latest
   ```

3. Test if your image works locally with the tira python utility. 

   ```
   pip install tira
   cd pan-code/clef23/trigger-detection
   tira-run --image <dockerhub-user>/pan23-trigger-detection-baseline-xgboost:latest \
      --input-directory ${PWD}/sample-data/input \
      --command 'python3 /baseline/baseline-xgboost-runner.py --input-dataset-dir $inputDataset --output-dir $outputDir'
   ```

4. Create a new docker software on TIRA (follow tira's [getting started](https://www.tira.io/t/getting-started/1364) guides for more details.).
   1. Tag and upload the image tag into your TIRA repository (follow the instructions under *Upload Images*
    
      ```
      docker login -u tira-user-<tira-group-name> -p <registry-password> registry.webis.de
      docker tag <dockerhub-user>/pan23-trigger-detection-baseline-xgboost:latest \
            registry.webis.de/code-research/tira/tira-user-<tira-group-name>/pan23-trigger-detection-baseline-xgboost:0.0.1
      ```
   
   2. Add a new Software under the *ADD CONTAINER* tab:
      
      **Command**: 
      
      `python3 /baseline/baseline-xgboost-runner.py --input-dataset-dir $inputDataset --output-dir $outputDir`
      
      **Docker Image**: 
      
      `registry.webis.de/code-research/tira/tira-user-<tira-group-name>/pan23-trigger-detection-baseline-xgboost:0.0.1`
      
      Click on the *ADD CONTAINER* button
      
  3. Go to the newly create tab. Select the resources for execution and the dataset for the run, then click *run container*. 
      
