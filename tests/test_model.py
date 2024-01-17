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
    sentences = ["This is an example sentence", "This is a fucking stupid sentence"]
    labels = torch.tensor([0, 1], dtype=torch.float32)
    torch.save(sentences, temp_data_dir / "train_data.pt")
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
    
    # Load sample data
    sentences = torch.load(sample_data / "train_data.pt")
    labels = torch.load(sample_data / "train_labels.pt")
    
    dataset = TensorDataset(model.embedder.encode(sentences, convert_to_tensor=True), labels)
    dataloader = DataLoader(dataset, batch_size=1)
    
    # Ensure the validation step works
    batch = next(iter(dataloader))
    loss = model.validation_step(batch)
    assert isinstance(loss, torch.Tensor)

    '''
    model = HatespeechClassification()
    sentences, labels = sample_data
    #Load data
    train_data = torch.load(fast_train_data_fpath)
    test_data = torch.load(fast_test_data_fpath)
    
    #dataset = TensorDataset(model.embedder.encode(sentences, convert_to_tensor=True), labels)
    dataloader = DataLoader(dataset, batch_size=1)
    
    # Ensure the validation step works
    batch = next(iter(dataloader))
    loss = model.validation_step(batch)
    assert isinstance(loss, torch.Tensor)
    '''

def test_optimizers():
    model = HatespeechClassification() # Do we need to test for all types of models?
    optimizer = model.configure_optimizers()
    
    # Ensure optimizer is of the correct type
    assert isinstance(optimizer, torch.optim.Optimizer)