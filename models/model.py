# Imports for the model
import torch
from sentence_transformers import SentenceTransformer

# Vi kan overveje at bruge Pythorch Lightnings 'LightningModule' til at definere modellen?
class MyModel:
    def __init__(self, model_name='aditeyabaral/sentencetransformer-bert-base-cased'):
        self.model = SentenceTransformer(model_name)

    def encode_sentences(self, sentences):
        embeddings = self.model.encode(sentences)
        return embeddings
    

# A simpple test to see that the model is loaded correctly:
sentences = ["This is an example sentence", "Each sentence is converted"]

my_model = MyModel()

embeddings = my_model.encode_sentences(sentences)

print(embeddings)
