import os
import click
import csv
from sentence_transformers import SentenceTransformer
import torch
import pandas as pd

RAW_DATA_FILENAME = "data/raw/HateSpeechDataset.csv"
PROCESSED_DATA_PATH = "data/processed"

<<<<<<< HEAD:02476 Hatespeech Classification/data/make_dataset.py
model_name = 'aditeyabaral/sentencetransformer-bert-base-cased'
=======
def check_if_dataset_exists(full_path):
    return os.path.exists(full_path)

def get_kaggle_credentials(kagglefile):
    if 'KAGGLE_USERNAME' in os.environ.keys() and 'KAGGLE_KEY' in os.environ.keys():
        print("Kaggle credentials already set")
    # If kagglefile does not exist, create it
    else:
        if not os.path.exists(kagglefile):
            print("kagglefile does not exist")
            print("please specify kaggle credencials username:")
            username = input()
            print("please specify kaggle credencials key:")
            key = input()
            api_key = {"username": username, "key": key}
            if not os.path.exists(os.path.dirname(kagglefile)):
                os.makedirs(os.path.dirname(kagglefile))
            f = open(kagglefile, "w")
            json.dump(api_key, f)
            f.close()
        # If kagglefile exists, read it
        else:
            print("kagglefile exists")
            f = open(kagglefile, "r")
            api_key = json.load(f)
            f.close()

    # set kaggle credentials
    os.environ['KAGGLE_USERNAME'] = api_key["username"]
    os.environ['KAGGLE_KEY'] = api_key["key"]
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()

    return api

def unzip_file(zip_file_path, extract_to_path):
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to_path)


def download_dataset(dataset_full_name, dataset_folder, kagglefile = "./data/kaggle.json"):
    kagglefile = "./data/kaggle.json"
    # get kaggle credentials
    api = get_kaggle_credentials(kagglefile)

    # download the dataset
    # save it to the specified path
    # if folder does not exist, create it:
    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)
    api.dataset_download_files(dataset_full_name, path=dataset_folder, unzip=True, quiet=False)
>>>>>>> f87c0913b301a092e8a2f7f7d1660fb1fadca310:hatespeech_classification_02476/data/make_dataset.py

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
