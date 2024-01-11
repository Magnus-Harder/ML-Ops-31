from pytorch_lightning import Trainer
from models.model import HatespeechClassification
import torch
import hydra
import logging

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def train(cfg):
    log.info(cfg)
    hparams = cfg["experiments"]

    # Create the model and trainer
    model = HatespeechClassification(**hparams["class_configuration"])
    trainer = Trainer(max_epochs=hparams["training_configuration"]["max_epochs"], accelerator="cpu", logger=log)

    # Load training data
    train_data = torch.load("data/processed/train_data.pt")
    train_labels = torch.load("data/processed/train_labels.pt")
    train_labels = train_labels[: len(train_data)].long()

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
        torch.utils.data.TensorDataset(train_data, train_labels), batch_size=batch_size
    )
    val_dataloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(val_data, val_labels), batch_size=batch_size
    )

    # Train
    trainer.fit(model, train_dataloader, val_dataloader)

    # Save model
    torch.save(model.state_dict(), "models/model.pt")


if __name__ == "__main__":
    train()
