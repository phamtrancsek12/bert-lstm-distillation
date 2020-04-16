import torch
from torchtext import data
from model.lstm import BiLSTM
from train import distill
from config import  EMBEDDING_PATH, DEVICE
from utils.load_data import load_distill_data
from utils.word2vec import load_pretrained_embedding
from utils.utils import set_seed
from utils.log import get_logger, out_dir
logger = get_logger(__file__.split("/")[-1])
set_seed()

def main():

    #  Load data
    text_field = data.Field(lower=True)
    label_field_1 = data.Field(sequential=False, use_vocab=False, dtype=torch.float)
    label_field_2 = data.Field(sequential=False, use_vocab=False, dtype=torch.float)
    train_iter, test_iter = load_distill_data(text_field, label_field_1, label_field_2)

    vocab_size = len(text_field.vocab)
    model = BiLSTM(vocab_size)
    model = model.to(DEVICE)

    # word_to_idx = text_field.vocab.stoi
    # pretrained_embeddings = load_pretrained_embedding(EMBEDDING_PATH, word_to_idx)
    # model.embeddings.weight.data.copy_(torch.from_numpy(pretrained_embeddings))

    distill(model, train_iter, test_iter)

if __name__ == "__main__":
    main()
