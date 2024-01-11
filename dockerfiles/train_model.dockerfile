# Base image
FROM python:3.11-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

WORKDIR /ML-Ops-31

COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY hatespeech_classification_02476/ hatespeech_classification_02476/

RUN pip install -r requirements.txt --no-cache-dir

# To use dvc: 
#COPY .git .git
#COPY .dvc .dvc
#COPY data.dvc data.dvc
#RUN dvc pull

# To just copy from folder:
COPY data/ data/

ENTRYPOINT ["python", "-u", "hatespeech_classification_02476/train_model.py"]