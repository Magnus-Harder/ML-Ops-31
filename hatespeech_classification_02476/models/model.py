# Imports for the model
import torch
from torch import nn, optim, exp
from sentence_transformers import SentenceTransformer
from pytorch_lightning import LightningModule
from torchmetrics import Accuracy
from torchmetrics.classification import BinaryStatScores


# Define LightningModule for Hatespeech Classification
class HatespeechClassification(LightningModule):
    def __init__(
        self, model_type="Fast", hidden_dim=128, activation="relu", dropout=0.4, learning_rate=1e-4, optimizer="Adam"
    ):
        super().__init__()

        # Define
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.accuracy = Accuracy(task="binary")
        self.binary_statscores = BinaryStatScores(threshold=0.5)
        self.model_dict = {"Best": "all-mpnet-base-v2", "Fast": "all-MiniLM-L6-v2"}

        # Get activation function
        activation_dict = {"relu": nn.ReLU(), "leaky_relu": nn.LeakyReLU(), "sigmoid": nn.Sigmoid(), "tanh": nn.Tanh()}
        try:
            self.activation_func = activation_dict[activation]
        except:
            raise ValueError("Activation function not found choose from 'relu', 'leaky_relu', 'sigmoid' or 'tanh'")

        # Define Variables to Determine wheather model is extrapolating on user input
        self.extrapolation = False

        # Load the Sentence Transformer model
        try:
            self.embedder = SentenceTransformer(self.model_dict[model_type])
        except:
            raise ValueError("Model not found choose from 'Best' (allmpnet-base-v2) or 'Fast' (all-MiniLM-L6-v2)")

        # Get the embedding size
        embedding_size = self.embedder.get_sentence_embedding_dimension()

        # Define the Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embedding_size, hidden_dim),
            nn.Dropout(dropout),
            self.activation_func,
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

        # Define the loss function
        self.loss_fn = nn.MSELoss()

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
        loss = self.loss_fn(pred.squeeze(), target)
        acc = self.accuracy(pred.squeeze(), target)

        # Log the results
        self.log("train_loss", loss)
        self.log("train_acc", acc)

        return loss

    # Validation step for pytorch lightning
    def validation_step(self, batch):
        # Get the data and target
        data, target = batch

        # Get the predictions and calculate the loss
        pred = self(data)
        loss = self.loss_fn(pred.squeeze(), target)
        acc = self.accuracy(pred.squeeze(), target)

        # Get the stats
        stats = self.binary_statscores(pred.squeeze(), target)
        tp, fp, tn, fn = stats[0], stats[1], stats[2], stats[3]

        # Calculate the rates
        tp_rate = tp / (tp + fn)
        tn_rate = tn / (tn + fp)

        # Log the results
        self.log("val_loss", loss)
        self.log("val_acc", acc)
        self.log("val_tp", tp_rate)
        self.log("val_tn", tn_rate)

        return loss

    # Optimizer for pytorch lightning
    def configure_optimizers(self):
        # Define the optimizer, currently only Adam and SGD are supported
        optimizer_dict = {"Adam": optim.Adam, "SGD": optim.SGD}
        try:
            optimizer = optimizer_dict[self.optimizer]
        except:
            raise ValueError("Optimizer not found choose from 'Adam' or 'SGD'")

        return optimizer(self.parameters(), lr=self.learning_rate)


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

    loss = my_model.loss_fn(pred_scores.squeeze(), torch.tensor([0, 1]))

    # Test Prediction forward pass
    my_model.extrapolation = True
    pred_scores = my_model(sentences)
    print("Prediction scores:")
    print(pred_scores.shape)
