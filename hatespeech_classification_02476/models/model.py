# Imports for the model
from torch import nn, optim, exp
from sentence_transformers import SentenceTransformer
from pytorch_lightning import LightningModule
from torchmetrics import Accuracy
from torchmetrics.classification import MulticlassConfusionMatrix
accuracy = Accuracy(task="multiclass", num_classes=2)
confusionmatrix = MulticlassConfusionMatrix(num_classes=2)


# 'aditeyabaral/sentencetransformer-bert-base-cased'
class HatespeechClassification(LightningModule):
    def __init__(
        self, model_type="Fast", hidden_dim=128, activation="relu", dropout=0.4, learning_rate=1e-4, optimizer="Adam"
    ):
        super().__init__()
        self.learning_rate = learning_rate

        model_dict = {"Best": "all-mpnet-base-v2", "Fast": "all-MiniLM-L6-v2"}

        # Define Variables to Determine wheather model is extrapolating on user input
        self.extrapolation = False

        # Load the Sentence Transformer model
        self.embedder = SentenceTransformer(model_dict[model_type])

        # Get the embedding size
        embedding_size = self.embedder.get_sentence_embedding_dimension()

        # Define the Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embedding_size, hidden_dim), 
            nn.Dropout(dropout),
            nn.ReLU(), 
            nn.Linear(hidden_dim, 1), 
            nn.Sigmoid()
        )

        # Define the loss function
        self.loss_fn = nn.BCELoss

    # Forward pass
    def forward(self, x):
        # If Input is encoded, then we are training
        if self.extrapolation == False:
            output = self.classifier(x)
        else:
            # Encode the sentence and convert to probability
            output = self.classifier(self.embedder.encode(x, convert_to_tensor=True))

        return output

    # Training step for pytorch lightning
    def training_step(self, batch):
        # Get the data and target
        data, target = batch

        # Get the predictions and calculate the loss
        pred = self(data)
        loss = self.loss_fn(pred, target)
        acc = accuracy(pred, target)

        self.log("train_loss", loss)
        self.log("train_acc", acc)
        

        return loss
    
    def validation_step(self, batch):

        # Get the data and target
        data, target = batch

        # Get the predictions and calculate the loss
        pred = self(data)
        loss = self.loss_fn(pred, target)
        acc = accuracy(pred, target)

        # Get the predictions
        pred = pred.argmax(dim=1)
        confmatrix = confusionmatrix(pred, target)

        self.log("val_loss", loss)
        self.log("val_acc", acc)
        self.log("val_confmatrix", confmatrix)

        return loss

    # Optimizer for pytorch lightning
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-2)


if __name__ == "__main__":
    # A simple test to see that the model is loaded correctly:
    sentences = ["This is an example sentence", "Each sentence is converted"]

    my_model = HatespeechClassification()
    embeddings = my_model.embedder.encode(sentences, convert_to_tensor=True)

    print("Embeddings:")
    print(embeddings.shape)

    # Test Training forward pass
    pred_scores = my_model(embeddings)
    print("Training Prediction scores:")
    print(pred_scores.shape)

    # Test Prediction forward pass
    my_model.extrapolation = True
    pred_scores = my_model(sentences)
    print("Prediction scores:")
    print(pred_scores.shape)
