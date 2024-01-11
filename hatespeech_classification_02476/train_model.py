from pytorch_lightning import Trainer
<<<<<<< HEAD:02476 Hatespeech Classification/train_model.py
from models.model import HatespeechClassification
=======
from models.model import MyModel
>>>>>>> b2d05d8b222b896e1a85ee211e323bf053019d41:hatespeech_classification_02476/train_model.py

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
    