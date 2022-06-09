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


def CTG_loss(input_ids, input_attn, target_ids, target_attn, attn_idx, attention_parameters, model):

    attention_weights = attention_parameters(attn_idx)
    # logging.info(f"attentionweight:{attention_weights}")
    loss_vec = model.get_loss_vec(
        input_ids, input_attn, target_ids=target_ids, target_attn=target_attn)
    loss = torch.dot(attention_weights, loss_vec)/input_ids.shape[0]

    return loss


# normal loss
def my_loss(input_ids, input_attn, target_ids, target_attn, model):

    with torch.no_grad():
        loss_vec = model.get_loss_vec(
            input_ids, input_attn, target_ids=target_ids, target_attn=target_attn)
        loss = torch.mean(loss_vec)
    return loss


def my_loss2(input_ids, input_attn, target_ids, target_attn, model):

    loss_vec = model.get_loss_vec(
        input_ids, input_attn, target_ids=target_ids, target_attn=target_attn)
    loss = torch.mean(loss_vec)

    return loss


# for the calculation of classification loss on the augmented dataset
# define calc_loss_aug

def calc_loss_aug(input_syn_ids, input_syn_attn, w_model, v_model):
    output_ids = w_model.generate(input_syn_ids, num_beams=1)[:,1:].contiguous()
    # print(output_ids)
    att = (output_ids > 0.5).long()
    # print(att)
    w_logits = w_model(input_syn_ids, input_syn_attn, target_ids=output_ids, target_attn=att).logits  # TODO,forward_decoderinput
    '''
    [ 1674, 26114,  9458,     6,     3,   362,  2701,  2587, 12957,  1197,
            25363, 12458,  1818, 12957, 23020,     7,     1,     1,  1674,     1,
                1,     3,  1674,  1674,  1674,     0,  1674,  1674,  1674,  1674,
            1674,  1674,  1674,  1674,  1674,  1674,     3,  1674,  1674,  1674,
            1674,  1674,     0,  1674,  1674,  1674,  1674,  1674,  1674,  1674,
            1674,  1674,  1674,  1674,     3,  1674]]
    whether this w_logits, the ouput outside att will affect the gumble?
    '''
    # print(torch.max(w_logits,-1))
    softmax_w_logtis = torch.softmax(w_logits,-1)# bs,sentlen,vocabsize
    hard_w_logits = torch.argmax(softmax_w_logtis,-1) #bs,sentlen
    # w_soft_idx, _ = torch.max(w_logits, dim=-1, keepdims=True)
    one_hot = torch.zeros(
        softmax_w_logtis.shape[0], softmax_w_logtis.shape[1], softmax_w_logtis.shape[-1], device=torch.device('cuda:0'))

    hard_w_logits_onehot = one_hot.scatter_(-1, hard_w_logits.unsqueeze(-1), 1.).float().detach(
    ) + softmax_w_logtis - softmax_w_logtis.detach()  #bug here  w_soft_idx.sum() # TODO:otputid start with 0

    loss_syn = v_model.loss(input_syn_ids, input_syn_attn,
                            target_ids=hard_w_logits_onehot[:,:,:32100], target_attn=att)  # TODO：forward_decoderinput
    
    #.model.loss only calculate for the loss for the target which attn = 1,
    return loss_syn
