import torch
import click
import os
from models.model import HatespeechClassification
from google.cloud import storage

BUCKET_NAME = "mlops-31-data-bucket"
MODEL_FILE = "models/model.pt"

client = storage.Client()
bucket = client.get_bucket(BUCKET_NAME)
blob = bucket.get_blob(MODEL_FILE)
prediction_model = HatespeechClassification()

prediction_model.load_state_dict(torch.load(blob.download_as_string()))

def hatespeech_predict(request):
    """Run prediction for a given input string.
    """
    prediction_model = HatespeechClassification()
    prediction_model.load_state_dict(torch.load(blob))
    prediction_model.eval()
    prediction_model.extrapolation = True
    
    print(predictions)
    return predictions
if __name__ == "__main__":
    predict()
