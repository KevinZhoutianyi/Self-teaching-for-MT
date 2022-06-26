
import torch.nn as nn
import torch
import math
import torch.nn as nn
import torch
import math

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, SubsetRandomSampler
from transformers import T5Tokenizer, T5ForConditionalGeneration
from datasets import load_dataset,load_metric
import torch
import logging
import numpy as np
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import sys
import time
from transformers.optimization import Adafactor
import os
import gc

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len,  d_model)
        pe[:,  0::2] = torch.sin(position * div_term)
        pe[:,  1::2] = torch.cos(position * div_term)
        pe = pe.cuda()
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: [ batch_size,seq_len]
        """
        ret =  self.pe[:x[1]]
        return ret
class AvgrageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val*n #TODO:its just for W
        self.cnt += n
        self.avg = self.sum / self.cnt

def tokenize(text_data, tokenizer, max_length, padding = True):
    
    encoding = tokenizer(text_data, return_tensors='pt', padding=padding, truncation = True, max_length = max_length)

    input_ids = encoding['input_ids']
    
    attention_mask = encoding['attention_mask']
    
    return input_ids, attention_mask
def get_Dataset(dataset, tokenizer,max_length):
    train_sentence = [x['de'] for x in dataset]
    train_target = [x['en'] for x in dataset]

  
    model1_input_ids, model1_input_attention_mask = tokenize(train_sentence, tokenizer, max_length = max_length)
  
    model1_target_ids, model1_target_attention_mask = tokenize(train_target, tokenizer, max_length = max_length)
 
    train_data = TensorDataset(model1_input_ids, model1_input_attention_mask, model1_target_ids, model1_target_attention_mask)
   
    return train_data