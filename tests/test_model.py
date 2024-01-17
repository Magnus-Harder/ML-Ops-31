from tests import _PATH_DATA, _PROJECT_ROOT, _TEST_ROOT
import torch
import os
import pytest
from hatespeech_classification_02476.models.model import HatespeechClassification

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