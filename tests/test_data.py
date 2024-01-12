from tests import _PATH_DATA, _PROJECT_ROOT, _TEST_ROOT
import torch
import os
import pytest

NUM_TRAINING_SAMPLES = 352719
NUM_TEST_SAMPLES = 0 #TODO 
NUM_FEATURES = 768

train_data_fpath = os.path.join(_PATH_DATA,"processed","train_data.pt")
test_data_fpath = os.path.join(_PATH_DATA,"processed","test_data.pt")

def test_data_exists():

    #Check that data files exists
    print(train_data_fpath)
    assert os.path.isfile(train_data_fpath), "Training data does not exist"
    assert os.path.isfile(test_data_fpath), "Test data does not exist"


@pytest.mark.skipif(not (os.path.isfile(train_data_fpath) or  os.path.isfile(train_data_fpath)), reason="Either training or test data is missing")
def test_data_dimensions():

    #Load data
    train_data = torch.load(train_data_fpath)
    test_data = torch.load(test_data_fpath)

    #Check that data dimensions are correct

    #Training data
    assert train_data.shape[0] == NUM_TRAINING_SAMPLES, "Training data does not have the correct number of samples"
    assert train_data.shape[1] == NUM_FEATURES, "Training data does not have correct number of features"

    #Test data
    assert test_data.shape[0] == NUM_TEST_SAMPLES, "Test data does not have the correct number of samples"
    assert test_data.shape[1] == NUM_FEATURES, "Test data does not have correct number of features"