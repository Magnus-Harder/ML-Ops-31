from pytorch_lightning import Trainer
from model import MyModel

model = MyModel()
trainer = Trainer()
trainer.fit(model)
