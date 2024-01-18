import torch
import click
import os
from models.model import HatespeechClassification
import pandas as pd

@click.command()
@click.argument("model_file", required=False, default="models/model.pt")
@click.argument("data_format", required=True, default="string")
@click.argument("data", required=True)
def predict(model_file, data_format, data):
    """Run prediction for a given model and data.
    Args:
        model: model to use for prediction
        dataloader: dataloader with batches
    Returns
        Tensor of shape [N, d] where N is the number of samples and d is the output dimension of the model
    """
    prediction_model = HatespeechClassification()
    prediction_model.load_state_dict(torch.load(model_file))
    prediction_model.eval()
    prediction_model.extrapolation = True

    if data_format == "string":
        predictions = prediction_model(data) > 0.5

    elif data_format == "csv":
        text_df = pd.read_csv(data)
        text_data = text_df[text_df.columns[0]].values
        predictions = prediction_model(text_data) > 0.5


    print(predictions)
    return predictions

if __name__ == "__main__": 
    predict() 

