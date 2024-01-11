import os
import click
import csv
from sentence_transformers import SentenceTransformer
import torch
import pandas as pd

RAW_DATA_FILENAME = "data/raw/HateSpeechDataset.csv"
PROCESSED_DATA_PATH = "data/processed"

model_name = 'aditeyabaral/sentencetransformer-bert-base-cased'

@click.command()
#@click.argument("dataset_fullname", required=False)
#@click.argument("dataset_folder", required=False)
def make_dataset():

    raw_df = pd.read_csv(RAW_DATA_FILENAME)
    
    #Remove empty rows
    raw_df = raw_df[raw_df['Label'] != 'Label']

    text_data = raw_df['Content'].values
    labels_tensor = torch.Tensor([int(label) for label in raw_df['Label'].tolist()])
    print("Loading sentence transformer...")

    encoding_model = SentenceTransformer(model_name)
    print("Encoding data...")

    text_encodings_tensor = encoding_model.encode(text_data, convert_to_tensor = True)
    print("Saving data...")

    torch.save(text_encodings_tensor, os.path.join(PROCESSED_DATA_PATH,'data.pt'))
    torch.save(labels_tensor, os.path.join(PROCESSED_DATA_PATH,'labels.pt'))
    
if __name__ == '__main__':
    #Process data
    make_dataset()
