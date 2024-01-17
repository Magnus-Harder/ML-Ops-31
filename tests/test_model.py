from tests import _PATH_DATA, _PROJECT_ROOT, _TEST_ROOT
import torch
import os
import pytest
from torch.utils.data import DataLoader, TensorDataset
from hatespeech_classification_02476.models.model import HatespeechClassification

fast_train_data_fpath = os.path.join(_PATH_DATA,"processed", "all-MiniLM-L6-v2", "train_data.pt")
fast_test_data_fpath = os.path.join(_PATH_DATA,"processed","all-MiniLM-L6-v2", "test_data.pt")

@pytest.fixture
def sample_data(tmp_path):
    # Create a temporary directory for storing sample data
    temp_data_dir = tmp_path / "sample_data"
    temp_data_dir.mkdir()

    # Create and save sample data
    my_model = HatespeechClassification()
    sentences = ["This is an example sentence", "This is a fucking stupid sentence"]
    embeddings = my_model.embedder.encode(sentences, convert_to_tensor=True)
    labels = torch.tensor([0, 1], dtype=torch.float32)
    torch.save(embeddings, temp_data_dir / "train_data.pt")
    torch.save(labels, temp_data_dir / "train_labels.pt")

    # Return datapath
    return temp_data_dir
    #return sentences, labels

def test_model():

    #Load model
    fast_model = HatespeechClassification(model_type = "Fast")
    best_model = HatespeechClassification(model_type = "Best")

    sentences = ["This is an example sentence", "Each sentence is converted"]

    fast_embeddings = fast_model.embedder.encode(sentences, convert_to_tensor=True)
    best_embeddings = best_model.embedder.encode(sentences, convert_to_tensor=True)

    #Test dimensionality of embeddings
    assert fast_embeddings.shape == (2, 384), "Incorrect dimensionality of embeddings"
    assert best_embeddings.shape == (2, 768), "Incorrect dimensionality of embeddings"

    #Test training forward pass
    fast_input = torch.zeros(10, 384)
    best_input = torch.zeros(10, 768)
    fast_pred = fast_model(fast_input)
    best_pred = best_model(best_input)
    assert fast_pred.shape == torch.Size([10, 1]), "Incorrect dimensionality of \"Fast\" model training output"
    assert best_pred.shape == torch.Size([10, 1]), "Incorrect dimensionality of \"Best\" model training output"

    #loss = my_model.loss_fn(pred_scores.squeeze(), torch.tensor([0, 1]))

    #Test prediction forward pass
    fast_model.extrapolation = True
    best_model.extrapolation = True
    pred_scores = fast_model(sentences)
    best_scores = best_model(sentences)
    assert pred_scores.shape == torch.Size([2, 1]), "Incorrect dimensionality of \"Fast\" model prediction output"
    assert best_scores.shape == torch.Size([2, 1]), "Incorrect dimensionality of \"Best\" model prediction output"

    
def test_validation_step(sample_data):
    model = HatespeechClassification()
    model.eval()
    
    # Load sample data
    #sentences = torch.load(sample_data / "train_data.pt")
    #labels = torch.load(sample_data / "train_labels.pt")
    # Load sample training data
    val_data = torch.load(sample_data / "train_data.pt")
    val_labels = torch.load(sample_data / "train_labels.pt")


    # Create DataLoader
    val_dataset = TensorDataset(val_data, val_labels)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=True)

    # Ensure the validation step works
    batch = next(iter(val_loader))
    loss = model.validation_step(batch)
    
    # Check if loss is a scalar tensor
    assert torch.is_tensor(loss) and loss.dim() == 0, "Training step did not return a scalar tensor"


def test_training_step(sample_data):
    # Load model
    model = HatespeechClassification()
    model.train()

    # Load sample training data
    train_data = torch.load(sample_data / "train_data.pt")
    train_labels = torch.load(sample_data / "train_labels.pt")

    # Create DataLoader
    train_dataset = TensorDataset(train_data, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

    # Run training step
    for batch in train_loader:
        loss = model.training_step(batch)

    # Check if loss is a scalar tensor
    assert torch.is_tensor(loss) and loss.dim() == 0, "Training step did not return a scalar tensor"
    
def test_optimizers():
    model = HatespeechClassification() # Do we need to test for all types of models?
    optimizer = model.configure_optimizers()
    
    # Ensure optimizer is of the correct type
    assert isinstance(optimizer, torch.optim.Optimizer)