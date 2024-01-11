from pytorch_lightning import Trainer
from models.model import MyModel

model = MyModel()
trainer = Trainer()
trainer.fit(model)
