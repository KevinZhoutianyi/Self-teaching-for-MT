import os
import gc
import random
import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from utils import *

import logging
def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
seed_torch(seed_)
from transformers import T5Tokenizer
tokenizer = T5Tokenizer.from_pretrained("t5-small")
# the loss for the encoder and the decoder model 
# this takes into account the attention for all the datapoints for the encoder-decoder model
def CTG_loss(input_ids, input_attn, target_ids, target_attn, attn_idx, attention_parameters, model):
    
    attention_weights = attention_parameters(attn_idx)
    # logging.info(f"attentionweight:{attention_weights}")
    loss_vec = model.get_loss_vec(input_ids, input_attn, target_ids = target_ids, target_attn = target_attn)
    loss = torch.dot(attention_weights, loss_vec)
    scaling_factor = 1
    
    return scaling_factor*loss


# normal loss
def my_loss(input_ids, input_attn, target_ids, target_attn, model):
    
    with torch.no_grad():
        loss_vec = model.get_loss_vec(input_ids, input_attn, target_ids = target_ids, target_attn = target_attn)
        
        loss = torch.mean(loss_vec,dim=0)
    
    return loss

def my_loss2(input_ids, input_attn, target_ids, target_attn, model):
    
    loss_vec = model.get_loss_vec(input_ids, input_attn, target_ids = target_ids, target_attn = target_attn)
    loss = torch.mean(loss_vec,dim=0)

    return loss
 
 

# for the calculation of classification loss on the augmented dataset
# define calc_loss_aug

def calc_loss_aug(input_syn_ids, input_syn_attn, w_model, v_model):
    output_ids = w_model.generate(input_syn_ids)
    #torch.Size([4, 132])print('input_syn_ids.shape',input_syn_ids.shape)
    #torch.Size([4, 132])print('input_syn_attn.shape',input_syn_attn.shape)
    #torch.Size([4, 53])print('output_ids.shape',output_ids.shape)
    att = (output_ids>0.5).long()
    #orch.Size([4, 53, 32128])print('att.shape',att.shape)
    w_logits = w_model(input_syn_ids, input_syn_attn, target_ids = output_ids, target_attn = torch.ones_like(output_ids).long()).logits
    w_soft_idx, bart_idx = torch.max(w_logits, dim=-1, keepdims= True)
    
    #torch.Size([4, 53, 32128])print('w_logits.shape',w_logits.shape)
    
    #torch.Size([4, 53, 1])print('w_soft_idx.shape',w_soft_idx.shape)
    one_hot = torch.zeros(output_ids.shape[0], output_ids.shape[1], v_model.vocab_size).cuda()
    w_output_ids = one_hot.scatter_(-1, output_ids.unsqueeze(-1), 1.).float().detach() + w_soft_idx.sum() - w_soft_idx.sum().detach()
    temp = torch.argmax(w_output_ids,-1)
    # print('temp',temp)
    # print('output_ids',output_ids)
    loss_syn = v_model.loss( input_syn_ids ,input_syn_attn   , target_ids = w_output_ids, target_attn = att)

    #[4, 53, 32100])print('one_hot.shape',one_hot.shape)
    #[4, 53, 32100])print('w_output_ids.shape',w_output_ids.shape)
    #print('loss_syn',loss_syn)
    # loss_vec = v_model.get_loss_vec(input_ids, input_attn, target_ids = output_v_ids, target_attn = output_v_attn)
    # loss = torch.sum(loss_vec,dim=0)
    del  w_output_ids, one_hot
        
    gc.collect()  
    

    return loss_syn