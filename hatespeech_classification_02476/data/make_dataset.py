import os
import click
from sentence_transformers import SentenceTransformer
import torch
import pandas as pd
from tqdm import tqdm

RAW_DATA_FILENAME = "data/raw/HateSpeechDataset.csv"
PROCESSED_DATA_PATH = "data/processed"

model_dict = {"Best": "all-mpnet-base-v2", "Fast": "all-MiniLM-L6-v2"}

@click.command()
@click.argument("training_ratio", required=False, default=0.8)
@click.argument("model_name", required=False, default="Fast")
def make_dataset(training_ratio, model_name):
    torch.manual_seed(0)

    raw_df = pd.read_csv(RAW_DATA_FILENAME)

    # Remove empty rows
    raw_df = raw_df[raw_df["Label"] != "Label"]

    text_data = raw_df["Content"].values
    labels_tensor = torch.Tensor([int(label) for label in raw_df["Label"].tolist()])

    # Embed text data and change model to mps
    print("Loading sentence transformer...")
    print("Model:", model_dict[model_name])
    encoding_model = SentenceTransformer(model_dict[model_name])

    # Change model to MPS
    if torch.backends.mps.is_available():
        print("Using MPS")
        device = torch.device("mps")
        encoding_model.to("mps")
    elif torch.cuda.is_available():
        print("Using CUDA")
        device = torch.device("cuda")
        encoding_model.to("cuda")
    else:
        print("Using CPU")
        device = torch.device("cpu")
        encoding_model.to("cpu")

    print("Encoding data...")
    text_encodings_tensor = torch.zeros(
        (len(labels_tensor), encoding_model.get_sentence_embedding_dimension()), device=device
    )

    # Loop over sentences in batches
    batch_size = 256

    for i in tqdm(range(0, len(text_data), batch_size)):
        if i + batch_size > len(text_data):
            batch_size = len(text_data) - i
        batch = text_data[i : i + batch_size]
        text_encodings_tensor[i : i + batch_size, :] = encoding_model.encode(batch, convert_to_tensor=True)

    # Shuffle data
    perm = torch.randperm(labels_tensor.shape[0])
    text_encodings_tensor = text_encodings_tensor[perm, :]
    labels_tensor = labels_tensor[perm]

    # Split data
    n_split = int(training_ratio * labels_tensor.shape[0])
    # print(n_split)

    train_data = text_encodings_tensor[:n_split]
    test_data = text_encodings_tensor[n_split:]
    train_labels = labels_tensor[:n_split]
    test_labels = labels_tensor[n_split:]

    # Move data to CPU
    train_data = train_data.to("cpu")
    test_data = test_data.to("cpu")
    train_labels = train_labels.to("cpu")
    test_labels = test_labels.to("cpu")

    print("Saving data...")
    torch.save(train_data, os.path.join(f"{PROCESSED_DATA_PATH}/{model_dict[model_name]}", "train_data.pt"))
    torch.save(test_data, os.path.join(f"{PROCESSED_DATA_PATH}/{model_dict[model_name]}", "test_data.pt"))
    torch.save(train_labels, os.path.join(f"{PROCESSED_DATA_PATH}/{model_dict[model_name]}", "train_labels.pt"))
    torch.save(test_labels, os.path.join(f"{PROCESSED_DATA_PATH}/{model_dict[model_name]}", "test_labels.pt"))


if __name__ == "__main__":
    # Process data
    make_dataset()
