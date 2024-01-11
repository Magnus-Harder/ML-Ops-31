# Imports for the model
from torch import nn, optim
from sentence_transformers import SentenceTransformer
from pytorch_lightning import LightningModule

class MyModel(LightningModule):
    def __init__(self, model_name='aditeyabaral/sentencetransformer-bert-base-cased'):

        super().__init__()

        self.embedder = SentenceTransformer(model_name)

        self.classifier = nn.Sequential(
            nn.Linear(768,128),
            nn.ReLU(),
            nn.Linear(128,2)
        )

        self.loss_fn = nn.CrossEntropyLoss()

    def encode_sentences(self, sentences):
        embeddings = self.embedder.encode(sentences)
        return embeddings
    
    def forward(self, x):
        if self.train == True:
            return self.classifier(x)
        else:
            return self.classifier(self.embedder.encode(x))


        return self.classifier(self.embedder.encode(x, convert_to_tensor = True))
    
    def training_step(self,batch):
        data, target = batch
        pred = self(data)
        loss = self.loss_fn(pred, target)
        return loss
    
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr = 1e-2)


if __name__ == '__main__':
    # A simple test to see that the model is loaded correctly:
    sentences = ["This is an example sentence", "Each sentence is converted"]

    my_model = MyModel()

    embeddings = my_model.encode_sentences(sentences)

    print(embeddings)
    print(embeddings.shape)

    #Test that model troughput gives sensible results
    pred_scores = my_model(sentences)
    print(pred_scores)
    print(pred_scores.shape)