from tests import _PATH_DATA, _PROJECT_ROOT, _TEST_ROOT
import torch
import os
import pytest

NUM_TRAINING_SAMPLES = 352719
NUM_TEST_SAMPLES = 88180
FAST_NUM_FEATURES = 384
#BEST_NUM_FEATURES = 

fast_train_data_fpath = os.path.join(_PATH_DATA,"processed", "all-MiniLM-L6-v2", "train_data.pt")
fast_test_data_fpath = os.path.join(_PATH_DATA,"processed","all-MiniLM-L6-v2", "test_data.pt")



def test_data_exists():

    #Check that data files exists
    assert os.path.isfile(fast_train_data_fpath), "Training data does not exist"
    assert os.path.isfile(fast_test_data_fpath), "Test data does not exist"


@pytest.mark.skipif(not (os.path.isfile(train_data_fpath) or  os.path.isfile(train_data_fpath)), reason="Either training or test data is missing")
def test_data_dimensions():

    #Load data
    train_data = torch.load(fast_train_data_fpath)
    test_data = torch.load(fast_test_data_fpath)

    #Check that data dimensions are correct

    #Training data
    assert train_data.shape[0] == NUM_TRAINING_SAMPLES, "Training data does not have the correct number of samples"
    assert train_data.shape[1] == FAST_NUM_FEATURES, "Training data does not have correct number of features"

    #Test data
    assert test_data.shape[0] == NUM_TEST_SAMPLES, "Test data does not have the correct number of samples"
    assert test_data.shape[1] == FAST_NUM_FEATURES, "Test data does not have correct number of features"