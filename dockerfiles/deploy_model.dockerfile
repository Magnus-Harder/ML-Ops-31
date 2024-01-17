FROM python:3.11-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

WORKDIR /ML-Ops-31

COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY hatespeech_classification_02476/ hatespeech_classification_02476/
COPY models/ models/

RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir reports

CMD ["uvicorn","--port", "8000", "hatespeech_classification_02476.app:app"]