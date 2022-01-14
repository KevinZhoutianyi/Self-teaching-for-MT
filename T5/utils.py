
import os
import torch
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from MT_hyperparams import *

def tokenize(text_data, tokenizer, max_length, padding = True):
    
    encoding = tokenizer(text_data, return_tensors='pt', padding=padding, truncation = True, max_length = max_length)

    input_ids = encoding['input_ids']
    
    attention_mask = encoding['attention_mask']
    
    return input_ids, attention_mask


def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
seed_torch(seed_)

def get_train_Dataset(dataset, tokenizer):
    # split half of the train to x and u
    
    # get the training data
    train_sentence = [x['en'] for x in dataset]
    train_target = [x['fr'] for x in dataset]
    n = len(train_sentence)
    
    # ratio = 0.3 # ratio of num of u/x tensor should be the same shape, cant seperate them here
    n = n//2*2
    attn_idx = torch.arange(n//2)     
    # print(attn_idx)#tensor([   0,    1,    2,  ..., 7575, 7576, 7577])
    ########################################################################################################
    
    # tokenize the article using the bart tokenizer
    model1_input_ids, model1_input_attention_mask = tokenize(train_sentence[:n//2], tokenizer, max_length = article_length)
    # print("Input shape: ")
    # print(model1_input_ids.shape, model1_input_attention_mask.shape)
    
    # tokenize the target using the bart tokenizer
    model1_target_ids, model1_target_attention_mask = tokenize(train_target[:n//2], tokenizer, max_length = summary_length)
    # print("Target shape: ")
    # print(model1_target_ids.shape, model1_target_attention_mask.shape)    

    ########################################################################################################
    
    # tokenize the target using the gpt tokenizer
    model2_input_raw_input_ids, model2_input_raw_attention_mask = tokenize(train_sentence[-n//2:], tokenizer, max_length = article_length)
    # print("Input shape: ")
    # print(model2_input_raw_input_ids.shape, model2_input_raw_attention_mask.shape)    


    # turn to the tensordataset
    train_data = TensorDataset(model1_input_ids, model1_input_attention_mask, model1_target_ids, model1_target_attention_mask, 
        model2_input_raw_input_ids, model2_input_raw_attention_mask, attn_idx)
   
    return train_data


def get_aux_dataset(dataset, tokenizer):
    
    
    # get the validing data
    aux_sentence = [x['en'] for x in dataset]
    aux_target = [x['fr'] for x in dataset]


    
    # tokenize the article using the bart tokenizer
    model1_input_ids, model1_input_attention_mask = tokenize(aux_sentence, tokenizer, max_length = article_length)
    # print("Input shape: ")
    # print(model1_input_ids.shape, model1_input_attention_mask.shape)
    
    # tokenize the target using the bart tokenizer
    model1_target_ids, model1_target_attention_mask = tokenize(aux_target, tokenizer, max_length = summary_length)
    # print("Target shape: ")
    # print(model1_target_ids.shape, model1_target_attention_mask.shape)    

    ########################################################################################################


    # turn to the tensordataset
    aux_data = TensorDataset(model1_input_ids, model1_input_attention_mask, model1_target_ids, model1_target_attention_mask)
   
    return aux_data

class AvgrageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt
def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    correct = correct.contiguous()

    res = []
    
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0/batch_size))
        
    return res