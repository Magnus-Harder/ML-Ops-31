from fastapi import FastAPI
from hatespeech_classification_02476.models.model import HatespeechClassification
import torch
from google.cloud import storage
from http import HTTPStatus
import pickle

app = FastAPI()

# Define data path on mounted GCS bucket
BUCKET_NAME = "ml-ops-data"
MODEL_FILE = 'models/model_online_exp2.pt'
HPARAMS_FILE = 'models/hparams_online_exp2.pkl'

# Load hparams
client = storage.Client()
bucket = client.bucket(BUCKET_NAME)

blob = bucket.blob(HPARAMS_FILE)
blob.download_to_filename('temp/hparams.pkl')

with open('temp/hparams.pkl', 'rb') as f:
    hparams = pickle.load(f)

# Load model class
prediction_model = HatespeechClassification(**hparams['class_configuration'])

# Load model weights
blob = bucket.blob(MODEL_FILE)
blob.download_to_filename('temp/model.pt')
state_dict = torch.load('temp/model.pt')
prediction_model.load_state_dict(state_dict)

# Set model to eval mode and extrapolation mode (for inference on unseen sentences)
prediction_model.eval()
prediction_model.extrapolation = True

print("Finished loading model and hparams")

@app.get("/")
def root():
    """ Health check."""
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
    }
    return response

@app.get("/predict")
def predict(input_data: str):
    print("input_data: ", input_data)
    print("type(input_data): ", type(input_data))
    predictions = prediction_model(input_data) > 0.5
    print("predictions: ", predictions)
    return {"predictions": predictions.tolist()}

