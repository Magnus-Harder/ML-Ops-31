from fastapi import FastAPI
from hatespeech_classification_02476.models.model import HatespeechClassification
import torch
from google.cloud import storage
from http import HTTPStatus

app = FastAPI()
BUCKET_NAME = "mlops-31-data-bucket"
MODEL_FILE = "models/model.pt"

client = storage.Client()
bucket = client.get_bucket(BUCKET_NAME)
blob = bucket.get_blob(MODEL_FILE)
with open('models/tmp_model.pt', 'wb') as file_obj:
    blob.download_to_file(file_obj)

prediction_model = HatespeechClassification()
prediction_model.load_state_dict(torch.load('models/tmp_model.pt'))
prediction_model.eval()
prediction_model.extrapolation = True

@app.get("/")
def root():
    """ Health check."""
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
    }
    return response

@app.post("/predict")
def predict(input_data: str):
    print("input_data: ", input_data)
    print("type(input_data): ", type(input_data))
    predictions = prediction_model(input_data) > 0.5
    print("predictions: ", predictions)
    return {"predictions": predictions.tolist()}

