# Base image
FROM python:3.11-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY hatespeech_classification_02476/ hatespeech_classification_02476/
#COPY data/ data/

WORKDIR /
RUN pip install -r requirements.txt --no-cache-dir
#RUN pip install . --no-deps --no-cache-dir

COPY .git .git
COPY .dvc .dvc
COPY data.dvc data.dvc
RUN dvc pull


ENTRYPOINT ["python", "-u", "02476 Hatespeech Classification/train_model.py"]