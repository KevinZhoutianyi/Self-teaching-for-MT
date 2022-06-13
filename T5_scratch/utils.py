
import os
import torch
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from MT_hyperparams import *

def count_parameters_in_MB(model):
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e6
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
    print('get train data start')
    train_sentence = [x['en'] for x in dataset]
    train_target = [x[target_language] for x in dataset]

  
    model1_input_ids, model1_input_attention_mask = tokenize(train_sentence, tokenizer, max_length = max_length)
  
    model1_target_ids, model1_target_attention_mask = tokenize(train_target, tokenizer, max_length = max_length)
 
    train_data = TensorDataset(model1_input_ids, model1_input_attention_mask, model1_target_ids, model1_target_attention_mask)
    
    print('get train data end')
   
    return train_data


def get_aux_dataset(dataset, tokenizer):
    
    
    # get the validing data
    aux_sentence = [x['en'] for x in dataset]
    aux_target = [x[target_language] for x in dataset]

    model1_input_ids, model1_input_attention_mask = tokenize(aux_sentence, tokenizer, max_length = max_length)
    model1_target_ids, model1_target_attention_mask = tokenize(aux_target, tokenizer, max_length = max_length)
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


def unpadding(message):
    last_char = message[-1]
    if ord(last_char) < 32:
        return message.rstrip(last_char)
    else:
        return message



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
def getGPUMem(device):
    return torch.cuda.memory_allocated(device=device)*100/torch.cuda.max_memory_allocated(device=device)

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('t5-small')
def d(l):
    return tokenizer.batch_decode(l,skip_special_tokens=True)
def en(l):
    return tokenizer.tokenize(l,tokenizer,512,True)

def turnoff_dropout(m):
    if type(m) == torch.nn.Dropout:
        m.p = 0
def turnon_dropout(m):
    if type(m) == torch.nn.Dropout:
        m.p = 0.1
def print_model(model):
    for k in model.state_dict():
        print(k,model.state_dict()[k])

from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

class Scheduler(_LRScheduler):
    def __init__(self, 
                 optimizer: Optimizer,
                 dim_embed: int,
                 warmup_steps: int,
                 initlr: int,
                 last_epoch: int=-1,
                 verbose: bool=False) -> None:

        self.dim_embed = dim_embed
        self.warmup_steps = warmup_steps
        self.num_param_groups = len(optimizer.param_groups)
        self.init_lr = initlr

        super().__init__(optimizer, last_epoch, verbose)
        
    def get_lr(self) -> float:
        lr = calc_lr(self._step_count, self.dim_embed, self.warmup_steps, self.init_lr)
        return [lr] * self.num_param_groups 


def calc_lr(step, dim_embed, warmup_steps, lr):
    return dim_embed**(-0.5) * min(step**(-0.5), step * warmup_steps**(-1.5)) *lr

def save(x,name):
    torch.save(x,'./model/'+name+'.pt')