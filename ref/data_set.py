import os
import torch
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from datasets import load_dataset
from utils import *
from hyperparams import *
# TensorDataset
def tokenize(text_data, tokenizer, max_length, padding = True):
    
    encoding = tokenizer(text_data, return_tensors='pt', padding=padding, truncation = True, max_length = max_length)

    input_ids = encoding['input_ids']
    
    attention_mask = encoding['attention_mask']
    
    return input_ids, attention_mask
def get_train_Dataset(dataset, num_points,  bart_tokenizer):
    
    # # print("dataset:",dataset)
    
    # get the training data
    train_sentence = dataset['train']['article'][:num_points]
    train_target = dataset['train']['highlights'][:num_points]

    # attention indices for calculation of losses
    attn_idx = torch.arange(len(train_sentence))    #index of each data, a_i means the weight of the loss for i data    
    # print(attn_idx)
    ########################################################################################################
    
    # tokenize the article using the bart tokenizer
    article_input_ids, article_attention_mask = tokenize(train_sentence, bart_tokenizer, max_length = article_length)
    # print("Input shape: ")
    # print(article_input_ids.shape, article_attention_mask.shape)
    
    # tokenize the target using the bart tokenizer
    target_input_ids_bart, target_attention_mask_bart = tokenize(train_target, bart_tokenizer, max_length = summary_length)
    # print("Target shape: ")
    # print(target_input_ids_bart.shape, target_attention_mask_bart.shape)    

  

    # turn to the tensordataset
    train_data = TensorDataset(article_input_ids, article_attention_mask, target_input_ids_bart, target_attention_mask_bart, 
         attn_idx,)

    # # print(next(iter(loader)))

    return train_data