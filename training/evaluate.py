import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report
from utils.log import get_logger
logger = get_logger(__file__.split("/")[-1])

def evaluate(model, data):
    """ Evaluate model """
    # Evaluate mode
    logger.info('Evaluate model...')
    model.eval()
    # Init
    avg_loss = 0.0
    truth_res = []
    pred_res = []
    with torch.no_grad():
        for batch in tqdm(data):
            # label_0 is real labels
            sent, label = batch.text, batch.label_0

            truth_res += list(label.data)
            # Init
            model.batch_size = len(label.data)
            model.lstm_hidden = model.init_hidden()
            # Get prediction
            # Label is argmax of output tensor
            pred = model(sent)
            pred_label = pred.data.max(1)[1].numpy()
            pred_res += [x for x in pred_label]

    # Calculate performance
    truth_res = torch.tensor(truth_res)
    pred_res = torch.as_tensor(np.array(pred_res))
    acc = accuracy_score(truth_res, pred_res)
    report = classification_report(truth_res, pred_res)
    logger.info('Accuracy: {}'.format(acc*100))
    logger.info(report)
    return acc, report