# Import libraries
import torch
from pytorch_lightning import Trainer
from models.model import HatespeechClassification
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping
import hydra
import omegaconf
from google.cloud import storage
import pickle

@hydra.main(version_base=None, config_path="conf", config_name="config")
def train(cfg):
    # Set up logging
    wandb_logger=WandbLogger(log_model="all")
    wandb_logger.log_hyperparams(omegaconf.OmegaConf.to_container(cfg))

    # Configure hyperparameters and data
    hparams = cfg["experiments"]

    # Configure Device
    if torch.backends.mps.is_available():
        print("Using MPS")
        device = 'cpu'
    elif torch.cuda.is_available():
        print("Using CUDA")
        device = 'cuda'
    else:
        print("Using CPU")
        device = 'cpu'

    # Create the model and trainer
    model = HatespeechClassification(**hparams["class_configuration"])
    early_stopping = EarlyStopping('val_loss')
    trainer = Trainer(max_epochs=hparams["training_configuration"]["max_epochs"], 
                      accelerator=device, 
                      logger=wandb_logger, 
                      enable_checkpointing=False, 
                      callbacks=[early_stopping])

    # Load training data
    DATAPATH = f"data/processed/{model.model_dict[hparams['class_configuration']['model_type']]}"
    train_data = torch.load(f"{DATAPATH}/train_data.pt")
    train_labels = torch.load(f"{DATAPATH}/train_labels.pt")

    # Move data to device
    train_data = train_data.to(device)
    train_labels = train_labels.to(device)

    # Create validation data split
    validation_ratio = 0.1
    n_split = int(validation_ratio * train_labels.shape[0])
    val_data = train_data[:n_split]
    val_labels = train_labels[:n_split]
    train_data = train_data[n_split:]
    train_labels = train_labels[n_split:]

    # Create dataloaders
    batch_size = hparams["training_configuration"]["batch_size"]
    train_dataloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_data, train_labels), batch_size=batch_size, num_workers=4, persistent_workers=True,
    )
    val_dataloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(val_data, val_labels), batch_size=batch_size, num_workers=4, persistent_workers=True
    )

    # Train
    trainer.fit(model, train_dataloader, val_dataloader)

    # Save model
    torch.save(model.state_dict(), "models/model.pt")
    BUCKET_NAME = "ml-ops-data"
    MODEL_FILE = "models/model_online.pt"

    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(MODEL_FILE)
    blob.upload_from_filename("models/model.pt")
    # Save hyperparameters
    with open("models/hparams.pkl", "wb") as f:
        pickle.dump(hparams, f)
    
    blob = bucket.blob("models/hparams_online.pkl")

    blob.upload_from_filename("models/hparams.pkl")
    

if __name__ == "__main__":
    train()
