This folder contains all code, data and images used in preparing the extended lab overview paper for the authorship verification task at PAN 2020. The relevant files are the following:

### Data
- `submissions`: the folder containing the unaltered predictions by each submitted system for the test set.
- `datasets/pan20-authorship-verification-test/truth.jsonl`: the ground ground for the test problems.
- `img/*.svg`: SVG-files for the images used in the extended lab overview paper.

### Spreadsheets
- `metrics.xlsx`: the final performance of each system in a tabular format.
- `predictions.xlsx`: a tabular overview of all predictions for all submissions per text pair.
- `predictions_topic.xlsx`: same as `predictions.xlsx`, but with a column for the topical dissimilarity for each text pair, as measured by a simple NMF-based topic model.

### Code
    * `pan20-authorship-verification-baseline-compressor.zip`: code for the compression baseline
    * `pan20-verif-baseline.py`: code for the naive, first order baseline 
    * `pan20_verif_evaluator.py`: implementation of the four evaluation metrics considered.
    * `collect.ipynb`: script used for running the final evaluation.
    * `tm_preprocessing.ipynb`: script for POS-tagging the text pairs using Spacy and writing the filtered content words to plain text files.
    * tm_model.ipynb: code for fitting the topic model and calculating the topical dissimilarity between the pairs in the test set.
    * `analysis.ipynb`: code for the exploratory analysis of the result (presented in the extended overview paper).
    
    