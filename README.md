# bert-lstm-distillation
Distilling knowledge from BERT to LSTM model

## Why distillation
BERT or other Transformer-like architecture recently achieved state-of-the-art in most of NLP tasks. However, the model size is very huge so it's not practical to deploy to production. 

Knowlegde Distillation is a technique to extract to knowlege from big model and teach it to a light model but keep the performance almost the same.

## Project details
In this project, I'm are going to train an simple LSTM binary classifier by distillation method. 

A BERT model is first trained on labelled data set (label: 0/1). Then I use this model to score on a lot of unlabelled data and store the logits output.
Finally, I train an LSTM model using MSE error on those output. 

In this repo, I only focus on the LSTM distillation part and assume that the training data has been generate by BERT model beforehand. 

## How to use
- Update training config in `config.py`
- Run `run_training`
