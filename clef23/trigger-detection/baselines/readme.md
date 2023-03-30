# XGBoost Baseline for the shared task on Trigger Detection at PAN23

This baseline uses [Gradient Boosted Trees](xgboost.readthedocs.io) based on a tf-idf document vectors. 
The `baseline-xgboost-trainer.py` script trains and saves the model and can run a ablation study. 
The `baseline-xgboost-runner.py` script loads the saved model and makes predictions. 

### Usage

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

1. After training the model, as shown above, run:

    ```
   docker build -t <dockerhub-user>/pan23-trigger-detection-baseline-xgboost:latest -f dockerfile.xgb .
   ```

2. Upload the image to a public docker repository, i.e. dockerhub. 

3. Create a new docker software on TIRA (follow the tira *getting started* guides).
   1. Enter the  following image tag into TIRA
    
      ```
      <dockerhub-user>/pan23-trigger-detection-baseline-xgboost:latest
      ```
   
   2. Enter the following command in TIRA to execute the software:

      ```
      python3 /baseline/baseline-xgboost-runner.py --input-dataset-dir $inputDataset --output-dir $outputDir
      ```