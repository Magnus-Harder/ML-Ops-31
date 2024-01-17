FROM python:3.11-slim
WORKDIR /
COPY ./app_requirements.txt /app_requirements.txt

RUN pip install --no-cache-dir --upgrade -r /requirements.txt
COPY ./app /app

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]