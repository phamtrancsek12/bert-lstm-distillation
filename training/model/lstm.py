""" LSTM model for distillation"""
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from config import USE_GPU, BATCH_SIZE, MODEL_CONFIG, LABEL_SIZE


class BiLSTM(nn.Module):
    """Bi-LSTM model"""
    def __init__(self, vocab_size):
        super(BiLSTM, self).__init__()
        self.use_gpu = USE_GPU
        self.batch_size = BATCH_SIZE
        self.hidden_dim = MODEL_CONFIG["hidden_dim"]
        # Define model architecture
        self.embeddings = nn.Embedding(vocab_size, MODEL_CONFIG["embedding_dim"])
        self.lstm = nn.LSTM(input_size=MODEL_CONFIG["embedding_dim"], hidden_size=MODEL_CONFIG["hidden_dim"], bidirectional=True)
        self.hidden2dense = nn.Linear(MODEL_CONFIG["hidden_dim"] * 2, MODEL_CONFIG["relu_dim"])
        self.dropout = nn.Dropout(MODEL_CONFIG["drop_out"])
        self.dense2label = nn.Linear(MODEL_CONFIG["relu_dim"], LABEL_SIZE)
        # Init weights
        self.lstm_hidden = self.init_hidden()

    def init_hidden(self):
        """
        Init weights for LSTM layer
        :return: (hidden h, cell c)
        """
        if self.use_gpu:
            return (Variable(torch.zeros(2, self.batch_size, self.hidden_dim).cuda()),
                    Variable(torch.zeros(2, self.batch_size, self.hidden_dim).cuda()))
        else:
            return (Variable(torch.zeros(2, self.batch_size, self.hidden_dim)),
                    Variable(torch.zeros(2, self.batch_size, self.hidden_dim)))

    def forward(self, sentence):
        x = self.embeddings(sentence).view(len(sentence), self.batch_size, -1)
        lstm_out, self.lstm_hidden = self.lstm(x, self.lstm_hidden)
        y = F.relu(lstm_out[-1])
        y = self.hidden2dense(y)
        y = self.dropout(y)
        y = self.dense2label(y)
        return y
