import os
import random
import torch
import numpy as np
from MT_hyperparams import *
from utils import *

def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
seed_torch(seed_)

class attention_params(torch.nn.Module):# A and B
    def __init__(self, args):
        super(attention_params, self).__init__()
        self.model_en2de = (torch.load(args.model_name_teacher.replace('/','')+'.pt')).encoder
        self.model_de2en = (torch.load(args.model_name_de2en.replace('/','')+'.pt')).encoder
        for k in self.model_en2de.parameters():
            k.requires_grad=False
        for k in self.model_de2en.parameters():
            k.requires_grad=False
        self.linear1 = torch.nn.Linear(512*2, 512, bias=False)
        self.linear2 = torch.nn.Linear(512, 1, bias=False)
        self.linear1.require_grad = True
        self.linear2.require_grad = True
        
        self.relu =  torch.nn.ReLU()
        self.Sigmoid = torch.nn.Sigmoid()
        self.softmax = torch.nn.Softmax()
        # torch.nn.init.xavier_uniform(self.linear.weight)
        
    def forward(self, x, x_attn, y,y_attn):
        encoded_x = self.model_en2de(x,x_attn).last_hidden_state#bs,seqlen,hiddensize
        encoded_x = torch.sum(encoded_x,1)/torch.sum(x_attn,1,keepdim=True)
        encoded_y = self.model_de2en(y,y_attn).last_hidden_state#bs,seqlen,hiddensize
        encoded_y = torch.sum(encoded_y,1)/torch.sum(y_attn,1,keepdim=True)#bs,hiddensize
        weight = self.relu(self.linear1(torch.hstack((encoded_x,encoded_y))))#bs,1
        weight = self.linear2(weight)#bs,1
        # print(torch.squeeze(weight))
        return self.softmax(torch.squeeze(weight))*x.shape[0]
        # weight = 
        
        # return probs