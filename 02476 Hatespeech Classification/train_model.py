from pytorch_lightning import Trainer
from models.model import HatespeechClassification

import hydra
import logging
log = logging.getLogger(__name__)

@hydra.main(version_base=None,config_path='conf',config_name="config")
def train(cfg):
    log.info(cfg)
    hparams = cfg['experiments']

    # Create the model and trainer
    model = HatespeechClassification(**hparams['class_configuration'])
    trainer = Trainer(max_epochs=hparams['training_configuration']['max_epochs'])

    # Get Dataloaders
    # Train the model
    #trainer.fit(model, train_dataloader, val_dataloader)

if __name__ == '__main__':
    train()
    