from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
import datetime
import pkg_resources
import seaborn as sns
import time
import scipy.stats as stats
import gc
import re
import os
import operator
import sys
import logging
import shutil
import pickle
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
import warnings
warnings.filterwarnings(action='once')
from apex import amp
from sklearn import metrics
from sklearn import model_selection
from nltk.stem import PorterStemmer
from sklearn.metrics import roc_auc_score
from tqdm import tqdm, tqdm_notebook

from pytorch_pretrained_bert import BertConfig
from pytorch_pretrained_bert import convert_tf_checkpoint_to_pytorch
from pytorch_pretrained_bert import BertTokenizer, BertAdam
from pytorch_pretrained_bert.modeling import BertPreTrainedModel, BertModel


device=torch.device('cuda')
MAX_SEQUENCE_LENGTH = 220
batch_size = 512
Data_dir="../data"
BERT_MODEL_PATH = '../models/uncased_L-12_H-768_A-12/'

class BertForSequenceClassification(BertPreTrainedModel):

    def __init__(self, config, num_labels):
        super(BertForSequenceClassification, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(0.15)
        self.linear1 = nn.Linear(config.hidden_size, config.hidden_size//2)
        self.linear2 = nn.Linear(config.hidden_size, config.hidden_size//2)
        self.linear_out = nn.Linear(config.hidden_size//2, 1)
        self.linear_aux_out = nn.Linear(config.hidden_size//2, num_labels-1)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)


        h_conc_linear1  = self.dropout(F.relu(self.linear1(pooled_output)))
        h_conc_linear2  = self.dropout(F.relu(self.linear2(pooled_output)))

        result = self.linear_out(h_conc_linear1)
        aux_result = self.linear_aux_out(h_conc_linear2)
        logits = torch.cat([result, aux_result], 1)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logitsclass BertForSequenceClassification(BertPreTrainedModel):

    def __init__(self, config, num_labels):
        super(BertForSequenceClassification, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(0.15)
        self.linear1 = nn.Linear(config.hidden_size, config.hidden_size//2)
        self.linear2 = nn.Linear(config.hidden_size, config.hidden_size//2)

        self.linear_out = nn.Linear(config.hidden_size//2, 1)
        self.linear_aux_out = nn.Linear(config.hidden_size//2, num_labels-1)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)

        h_conc_linear1  = self.dropout(F.relu(self.linear1(pooled_output)))
        h_conc_linear2  = self.dropout(F.relu(self.linear2(pooled_output)))

        result = self.linear_out(h_conc_linear1)
        aux_result = self.linear_aux_out(h_conc_linear2)
        logits = torch.cat([result, aux_result], 1)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits


def convert_lines(example, max_seq_length,tokenizer):
    max_seq_length -=2
    all_tokens = []
    longer = 0
    for text in tqdm_notebook(example):
        tokens_a = tokenizer.tokenize(text)
        if len(tokens_a)>max_seq_length:
            tokens_a = tokens_a[:max_seq_length]
            longer += 1
        one_token = tokenizer.convert_tokens_to_ids(["[CLS]"]+tokens_a+["[SEP]"])+[0] * (max_seq_length - len(tokens_a))
        all_tokens.append(one_token)
    print(longer)
    return np.array(all_tokens)

os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

model = BertForSequenceClassification.from_pretrained("../working", cache_dir=None, num_labels=7)
model.cuda()
model = nn.DataParallel(model, device_ids=[0,1])
model.load_state_dict(torch.load("./epoch_" + str(0) + "_bert_pytorch_state.bin"))

for param in model.parameters():
    param.requires_grad = False
model.eval()
model.zero_grad()


test_df = pd.read_csv(os.path.join(Data_dir,"test.csv"))

tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_PATH, cache_dir=None,do_lower_case=True)
test_sequences = convert_lines(test_df["comment_text"].fillna("DUMMY_VALUE"),MAX_SEQUENCE_LENGTH, tokenizer)


test_preds = np.zeros((len(test_sequences)))
test = torch.utils.data.TensorDataset(torch.tensor(test_sequences, dtype=torch.long))
test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)

tk0 = tqdm_notebook(test_loader)
for i, (x_batch,) in enumerate(tk0):
    pred = model(x_batch.to(device), attention_mask=(x_batch>0).to(device), labels=None)
    test_preds[i*batch_size:(i+1)*batch_size] = pred[:,0].detach().cpu().squeeze().numpy()

test_preds_sigmoid = torch.sigmoid(torch.tensor(test_preds)).numpy()
submission = pd.DataFrame.from_dict({
    'id': test_df['id'],
    'prediction': test_preds_sigmoid
})
submission.to_csv('../pc/pred/pc_epoch_1_submission.csv', index=False)
