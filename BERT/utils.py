
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

def get_data(dataset, tokenizer):
    print('get train data start')
    batch = dataset['sentence']
    labels = torch.tensor(dataset['label'])

  
    encoding = tokenizer(batch, return_tensors='pt', padding=True, truncation = True, max_length=max_length)  
   
    
    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']
    
    # print the shapes
    print("Input shape: ")
    print(input_ids.shape, attention_mask.shape,labels.shape)
    
    # turn to the tensordataset
    train_data = TensorDataset(input_ids, attention_mask, labels)
    
    return train_data
    
def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0/batch_size))
        
    return res


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
    return tokenize(l,tokenizer,512,True)

def turnoff_dropout(m):
    if type(m) == torch.nn.Dropout:
        m.p = 0
def turnon_dropout(m):
    if type(m) == torch.nn.Dropout:
        m.p = 0.1
def save(x,name):
    torch.save(x,'./model/'+name+'.pt')

# def randomize_model(model):
#     for module_ in model.named_modules(): 
#         # print('!',type(module_[1]))
#         if isinstance(module_[1],(torch.nn.Linear, torch.nn.Embedding)):
#             module_[1].weight.data.normal_(mean=0.0, std=0.001)
#         elif isinstance(module_[1], transformers.models.t5.modeling_t5.T5LayerNorm):
#             module_[1].weight.data.fill_(0.1)
#         if isinstance(module_[1], torch.nn.Linear) and module_[1].bias is not None:
#             module_[1].bias.data.zero_()
#     return model
# from transformers import AutoTokenizer, GPT2LMHeadModel, T5ForConditionalGeneration, T5Config
# config = T5Config.from_pretrained('t5-small')
# print(config)
# model = T5ForConditionalGeneration(config)
# model = randomize_model(model)
# model_size = sum(t.numel() for t in model.parameters())
# print(f"T5 size: {model_size/1000**2:.1f}M parameters")
# modelname = 't5-small'
# torch.save(model,modelname+'.pt')
def compare_model(m1,m2):
    for k1,k2 in zip(m1.state_dict(),m1.state_dict()):
        print(k1[:30],'\t',torch.sum(m1.state_dict()[k1]-m2.state_dict()[k2]))
def load():
    import glob
    # All files and directories ending with .txt and that don't begin with a dot:
    l =  glob.glob("./model/*.pt")
    print(l)
    m = []
    for x in l:
        m.append(torch.load(x))
    return m