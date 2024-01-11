import os
import click
import requests

# froomp
import json
import zipfile

DEFAULT_DATASET = "waalbannyantudre/hate-speech-detection-curated-dataset/"
DEFAULT_FOLDER = "data/raw/"


def check_if_dataset_exists(full_path):
    return os.path.exists(full_path)


def get_kaggle_credentials(kagglefile):
    if "KAGGLE_USERNAME" in os.environ.keys() and "KAGGLE_KEY" in os.environ.keys():
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
    os.environ["KAGGLE_USERNAME"] = api_key["username"]
    os.environ["KAGGLE_KEY"] = api_key["key"]
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()

    return api


def unzip_file(zip_file_path, extract_to_path):
    with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
        zip_ref.extractall(extract_to_path)


def download_dataset(dataset_full_name, dataset_folder, kagglefile="./data/kaggle.json"):
    kagglefile = "./data/kaggle.json"
    # get kaggle credentials
    api = get_kaggle_credentials(kagglefile)

    # download the dataset
    # save it to the specified path
    # if folder does not exist, create it:
    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)
    api.dataset_download_files(dataset_full_name, path=dataset_folder, unzip=True, quiet=False)


@click.command()
@click.argument("dataset_fullname", required=False)
@click.argument("dataset_folder", required=False)
def make_dataset(dataset_fullname, dataset_folder):
    # either both parameters are specified or none:
    if (dataset_fullname == None) and (dataset_folder == None):
        dataset_fullname = DEFAULT_DATASET
        dataset_folder = DEFAULT_FOLDER
    elif (dataset_fullname == None) or (dataset_folder == None):
        raise ValueError("Both dataset_fullname and dataset_folder must be specified")
    download_dataset(dataset_fullname, dataset_folder)


if __name__ == "__main__":
    # Get the data and process it
    make_dataset()
