# Text Detoxification

The code in this directory is used to evaluate the predictions of the [text detoxification shared task at CLEF 2025](https://pan.webis.de/clef25/pan25-web/text-detoxification.html).

Here, you can find the code for evaluation and baselines.

# CodaLab baseline submission

For participation through [CodaLab platform](https://codalab.lisn.upsaclay.fr/competitions/22396), please refer to [data](sample_submissions/) folder to find the submission examples of the duplicated baseline.

## Baselines

We provide a set of baselines:

- [baseline-delete](baselines/baseline_delete/)
- [baseline-backtranslation](baselines/baseline_backtranslation/)
- [baseline-mt0](baselines/baseline_mt0/)
- [llm-baselines](baselines/openai/)

## Evaluation

### Requirements
Works for:
- `python 3.11.*`
- `torch==2.1.1+cu121`

### Evaluation Criteria

The submissions will be evaluated using different metrics:

1. **Similarity measurement:** Measuring similarity between generated outputs, golden outputs, and given inputs. Here we employ the [sentence-transformers/LaBSE](https://huggingface.co/sentence-transformers/LaBSE) model.
2. **Toxicity classifier:** Employing a toxicity classifier to measure how your generated outputs were detoxified compared to given inputs and golden outputs. Here we rely on the [textdetox/xlmr-large-toxicity-classifier-v2](https://huggingface.co/textdetox/xlmr-large-toxicity-classifier-v2) model.
3. **X-comet for Fluency:** Based on our experiments, COMET machine translation models showed perfect correlation with the annotation of fluency in detoxified texts. Especially, we rely on the [myyycroft/XCOMET-lite](https://huggingface.co/myyycroft/XCOMET-lite) model.


### How to evaluate your submission
```python

# see arguments with provided texts and parameters
python evaluation/evaluate.py
```

```
>>> [{'lang': 'am', 'STA': 1.0, 'SIM': 1.0, 'XCOMET': 0.9268987967570623, 'J': 0.9268987967570623}, {'lang': 'ar', 'STA': 1.0, 'SIM': 0.9999999998509884, 'XCOMET': 0.946625092625618, 'J': 0.9466250924830601}, {'lang': 'de', 'STA': 1.0, 'SIM': 0.9999999999006589, 'XCOMET': 0.9824136279026667, 'J': 0.9824136278044319}, {'lang': 'en', 'STA': 1.0, 'SIM': 1.0, 'XCOMET': 0.7565387411415577, 'J': 0.7565387411415577}, {'lang': 'es', 'STA': 1.0, 'SIM': 0.9999999999006589, 'XCOMET': 0.9624946667750677, 'J': 0.9624946666793917}, {'lang': 'fr', 'STA': 1.0, 'SIM': 1.0, 'XCOMET': 0.9662858235836029, 'J': 0.9662858235836029}, {'lang': 'he', 'STA': 1.0, 'SIM': 1.0, 'XCOMET': 0.9520459377765655, 'J': 0.9520459377765655}, {'lang': 'hi', 'STA': 1.0, 'SIM': 0.9999999997516473, 'XCOMET': 0.9553370489676793, 'J': 0.9553370487302921}, {'lang': 'hin', 'STA': 1.0, 'SIM': 0.9999999994039536, 'XCOMET': 0.9149222415685654, 'J': 0.9149222410194864}, {'lang': 'it', 'STA': 1.0, 'SIM': 1.0, 'XCOMET': 0.9616609334945678, 'J': 0.9616609334945678}, {'lang': 'ja', 'STA': 1.0, 'SIM': 1.0, 'XCOMET': 0.9549923813343049, 'J': 0.9549923813343049}, {'lang': 'ru', 'STA': 1.0, 'SIM': 0.9999999999006589, 'XCOMET': 0.9612591344118119, 'J': 0.961259134316393}, {'lang': 'tt', 'STA': 1.0, 'SIM': 1.0, 'XCOMET': 0.9360820281505585, 'J': 0.9360820281505585}, {'lang': 'uk', 'STA': 1.0, 'SIM': 1.0, 'XCOMET': 0.9617038138707479, 'J': 0.9617038138707479}, {'lang': 'zh', 'STA': 1.0, 'SIM': 0.9999999998013178, 'XCOMET': 0.9559485997756322, 'J': 0.9559485995835458}]
```
