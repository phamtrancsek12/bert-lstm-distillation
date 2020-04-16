""" Load train and test data from file """
import dill as pickle
import torch
from torchtext import data
from config import DATA_PATH, TRAIN_FILE, TEST_FILE, BATCH_SIZE, TEXT_FIELD_FILE
from utils.log import get_logger, out_dir
logger = get_logger(__file__.split("/")[-1])


def load_distill_data(text_field, label_field_1, label_field_2):
    logger.info("Load training data....")
    # Load data from csv file
    # For train data, labels are logits output from teacher model
    # For test data, labels are binary (0/1)
    train, test = data.TabularDataset.splits(path=DATA_PATH, train=TRAIN_FILE, test=TEST_FILE, format='csv',
                                                  fields=[('text', text_field), ('label_1', label_field_1), ('label_2', label_field_2)])

    # Build vocab
    text_field.build_vocab(train, test)

    # Create iterators
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_iter, test_iter = data.BucketIterator.splits((train, test),
                batch_sizes=(BATCH_SIZE, BATCH_SIZE, BATCH_SIZE),
                sort_key=lambda x: len(x.text), repeat=False, device=device, shuffle=True)

    # Save text_field and label_field for prediction
    with open(TEXT_FIELD_FILE.format(out_dir), 'wb') as f:
        pickle.dump(text_field, f)

    return train_iter, test_iter