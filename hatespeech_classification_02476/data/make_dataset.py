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
@click.argument("training_ratio", required=False, default = 0.8)
#@click.argument("dataset_folder", required=False)
def make_dataset(training_ratio):

    torch.manual_seed(0)

    raw_df = pd.read_csv(RAW_DATA_FILENAME)
    
    #Remove empty rows
    raw_df = raw_df[raw_df['Label'] != 'Label']

    text_data = raw_df['Content'].values
    labels_tensor = torch.Tensor([int(label) for label in raw_df['Label'].tolist()])

    #Embed text data
    print("Loading sentence transformer...")
    encoding_model = SentenceTransformer(model_name)
    print("Encoding data...")
    N = 1000
    text_encodings_tensor = encoding_model.encode(text_data[:N], convert_to_tensor = True)
    labels_tensor = labels_tensor[:N]
    #Shuffle data
    perm = torch.randperm(labels_tensor.shape[0])
    #print(text_encodings_tensor[:N,:].shape[0])
    text_encodings_tensor = text_encodings_tensor[perm,:]
    labels_tensor = labels_tensor[perm]
    #print(labels_tensor.shape)
    #print(text_encodings_tensor.shape)
    
    #Split data
    n_split = int(training_ratio * labels_tensor.shape[0])
    #print(n_split)
    
    train_data = text_encodings_tensor[:n_split]
    test_data = text_encodings_tensor[n_split:]
    train_labels = labels_tensor[:n_split]
    test_labels = labels_tensor[n_split:]

    #print(train_data.shape)
    #print(test_data.shape)
    #print(train_labels.shape)
    #print(test_labels.shape)

    print("Saving data...")
    torch.save(train_data, os.path.join(PROCESSED_DATA_PATH,'train_data.pt'))
    torch.save(train_data, os.path.join(PROCESSED_DATA_PATH,'test_data.pt'))
    torch.save(labels_tensor, os.path.join(PROCESSED_DATA_PATH,'train_labels.pt'))
    torch.save(labels_tensor, os.path.join(PROCESSED_DATA_PATH,'test_labels.pt'))
    
if __name__ == '__main__':
    #Process data
    make_dataset()
