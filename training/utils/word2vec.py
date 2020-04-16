""" Load word2vec embedding"""
import numpy as np
from utils.log import get_logger
logger = get_logger(__file__.split("/")[-1])

def load_bin_vec(fname, vocab):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    word_vecs = {}
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in range(vocab_size):
            word = []
            while True:
                ch = f.read(1).decode('latin-1')
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)
            if word in vocab:
               word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')
            else:
                f.read(binary_len)
    return word_vecs

def load_pretrained_embedding(embedding_path, word_to_idx):
    """ Load embedding """
    logger.info("Load WordEmbedding...\n")
    # Randomize the pretrained embeddings
    pretrained_embeddings = np.random.uniform(-0.25, 0.25, (len(word_to_idx), 300))
    pretrained_embeddings[0] = 0

    # Load pretrained from file
    if embedding_path != None:
        word2vec = load_bin_vec(embedding_path, word_to_idx)
        for word, vector in word2vec.items():
            pretrained_embeddings[word_to_idx[word]-1] = vector
    return pretrained_embeddings
