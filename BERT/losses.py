from transformers import T5Tokenizer
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
tokenizer = T5Tokenizer.from_pretrained("t5-small")
# the loss for the encoder and the decoder model
# this takes into account the attention for all the datapoints for the encoder-decoder model


def CTG_loss(input_ids, input_attn, target_ids, attention_parameters, attn_idx, model):

    attention_weights = attention_parameters(input_ids, input_attn,attn_idx)
    # logging.info(f"attentionweight:{attention_weights}")
    logits,loss_vec = model.get_loss_vec(
        input_ids,input_attn, target_ids)
    loss = torch.dot(attention_weights, loss_vec)/input_ids.shape[0]

    return logits,loss


# normal loss
def my_loss(input_ids, input_attn, target_ids, model):

    with torch.no_grad():
        logits,loss_vec = model.get_loss_vec(
        input_ids,input_attn, target_ids)
        loss = torch.mean(loss_vec)
    return logits,loss


def my_loss2(input_ids, input_attn, target_ids, model):

    logits,loss_vec = model.get_loss_vec(
        input_ids,input_attn, target_ids)
    loss = torch.mean(loss_vec)

    return logits,loss


# for the calculation of classification loss on the augmented dataset
# define calc_loss_aug

def calc_loss_aug(input_syn_ids, input_syn_attn, w_model, v_model):
    w_model.eval()
    output_ids = w_model(input_syn_ids,input_syn_attn)
    w_model.train()
    output_ids = torch.softmax(output_ids,-1)
    # target = soft_frequency(output_ids, probs = True, soft = True)
    loss_syn = torch.mean(v_model.get_loss_vec(input_syn_ids, input_syn_attn,output_ids)[1])
    
    # w_model.apply(turnon_dropout)
    #.model.loss only calculate for the loss for the target which attn = 1,
    return loss_syn


def soft_frequency( logits,  probs=False, soft = True):
    """
    Unsupervised Deep Embedding for Clustering Analysis
    https://arxiv.org/abs/1511.06335
    """
    power = 2
    if not probs:
        softmax = nn.Softmax(dim=1)
        y = softmax(logits.view(-1, logits.shape[-1])).view(logits.shape)
    else:
        y = logits
    f = torch.sum(y, dim=0)
    t = y**power / f
    #print('t', t)
    t = t + 1e-10
    p = t/torch.sum(t, dim=-1, keepdim=True)
    return p if soft else torch.argmax(p, dim=1)
def uncertainty(x,x_attn,model):
    logits = model(x,x_attn)
    p = torch.softmax(logits,-1)
    return entropy(p)
def entropy(p, dim = -1, keepdim = None):
   return -torch.where(p > 0, p * p.log(), p.new([0.0])).sum(dim = dim) # can be a scalar, when PyTorch.supports it