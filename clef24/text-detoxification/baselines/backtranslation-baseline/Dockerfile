FROM pytorch/pytorch:2.1.1-cuda12.1-cudnn8-devel

RUN apt-get update && apt-get install -y python3 python3-pip python-is-python3 git

COPY ./requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN python3 -c "from transformers import M2M100ForConditionalGeneration, BartForConditionalGeneration, AutoTokenizer; \
    M2M100ForConditionalGeneration.from_pretrained('facebook/nllb-200-distilled-600M'); \
    BartForConditionalGeneration.from_pretrained('s-nlp/bart-base-detox'); \
    AutoTokenizer.from_pretrained('facebook/nllb-200-distilled-600M'); \
    AutoTokenizer.from_pretrained('s-nlp/bart-base-detox');"

COPY backtranslation_baseline.py .
ADD backtranslation_baseline.py /backtranslation_baseline.py
RUN chown root:root /backtranslation_baseline.py && \
    chmod +x /backtranslation_baseline.py

CMD ["python3", "backtranslation_baseline.py"]

