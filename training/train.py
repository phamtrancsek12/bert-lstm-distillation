""" Train """
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from config import NUM_EPOCHS, LR, USE_GPU
from evaluate import evaluate
from utils.log import get_logger, out_dir
logger = get_logger(__file__.split("/")[-1])

def train_epoch(model, train_iter, loss_function, optimizer, epoch):
    """ Train one epoch """
    # Train mode
    model.train()
    # Init
    avg_loss = 0.0
    truth_res = []
    pred_res = []

    logger.info('Epoch {}'.format(epoch))
    for batch in tqdm(train_iter, desc='Train epoch '+str(epoch)):
        # Get sentence with logit values (from teacher model)
        sent, label_1, label_2 = batch.text, batch.label_1, batch.label_2
        target_logits = torch.stack((batch.label_1, batch.label_2), -1)

        # Label is argmax of logits
        if USE_GPU:
            truth_label = target_logits.max(1)[1].cpu().numpy()
        else:
            truth_label = target_logits.max(1)[1].numpy()
        truth_res += [x for x in truth_label]
        # Init
        model.batch_size = len(label_1.data)
        model.lstm_hidden = model.init_hidden()
        # Fit data
        pred = model(sent)
        # Get prediction label
        if USE_GPU:
            pred_label = pred.data.max(1)[1].cpu().numpy()
        else:
            pred_label = pred.data.max(1)[1].numpy()
        pred_res += [x for x in pred_label]
        # Do backprop
        model.zero_grad()
        loss = loss_function(pred, target_logits)
        avg_loss += loss.item()
        loss.backward()
        optimizer.step()

    # Loss and performance
    avg_loss /= len(train_iter)
    truth_res = torch.tensor(truth_res)
    pred_res = torch.as_tensor(np.array(pred_res))
    acc = accuracy_score(truth_res, pred_res)
    logger.info('Train: loss %.2f acc %.1f' % (avg_loss, acc*100))

def distill(model, train_iter, test_iter):
    """ Train model by distillation data """
    # Init
    best_dev_acc = 0.0
    optimizer = optim.Adam(model.parameters(), lr=LR)
    loss_function = nn.MSELoss() # According to the paper, using only logits loss work best

    for epoch in range(NUM_EPOCHS):
        train_epoch(model, train_iter, loss_function, optimizer, epoch)
        dev_acc, report = evaluate(model, test_iter)
        if dev_acc > best_dev_acc:
            # Save best model
            best_dev_acc = dev_acc
            logger.info('=> Saving model....')
            torch.save(model.state_dict(), '{}/best_model.pth'.format(out_dir))