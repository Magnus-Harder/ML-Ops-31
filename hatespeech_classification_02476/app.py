from fastapi import FastAPI
from models.model import HatespeechClassification
import torch
from google.cloud import storage

app = FastAPI()
model = HatespeechClassification()
BUCKET_NAME = "mlops-31-data-bucket"
MODEL_FILE = "models/model.pt"

client = storage.Client()
bucket = client.get_bucket(BUCKET_NAME)
blob = bucket.get_blob(MODEL_FILE)
with open('data/tmp_model.pt', 'wb') as file_obj:
    blob.download_to_file(file_obj)

prediction_model = HatespeechClassification()
prediction_model.load_state_dict(torch.load('data/tmp_model.pt'))
prediction_model.eval()
prediction_model.extrapolation = True
#model.load_state_dict(torch.load("models/model.pt"))

@app.post("/predict")
def predict(input_data: str):
    predictions = model(input_data) > 0.5
    return {"predictions": predictions.tolist()}