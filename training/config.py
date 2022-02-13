""" config """
import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_GPU = torch.cuda.is_available()
print("Devices: {}".format(DEVICE))

DATA_PATH = "./data/"
TRAIN_FILE = 'train.csv'
TEST_FILE = 'valid.csv'
TEXT_FIELD_FILE = '{}/text_field.pkl'
LABEL_FIELD_FILE = '{}/label_field.pkl'
EMBEDDING_PATH = "./data/GoogleNews-vectors-negative300.bin"

SEED = 1234
LR = 0.001
NUM_EPOCHS = 10
BATCH_SIZE = 8
LABEL_SIZE = 2

MODEL_CONFIG = {
    "embedding_dim": 300,
    "hidden_dim": 300,
    "relu_dim": 400,
    "drop_out": 0.3
}