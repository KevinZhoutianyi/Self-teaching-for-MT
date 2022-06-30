import os
import random
import torch
import numpy as np
from MT_hyperparams import *
from utils import *
from model import Model

def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
seed_torch(seed_)

class attention_params(torch.nn.Module):# A and B
    def __init__(self, tokenizer, args):
        super(attention_params, self).__init__()
        self.model = Model(tokenizer,args,'A').cuda()
        self.Sigmoid = torch.nn.Sigmoid()
        
        
    def forward(self, x, attn):
        weight = self.model(x,attn)
        weight = self.Sigmoid(weight[:,0])
        weight = torch.clamp(weight, min=0.1,max=0.9)
        return weight*x.shape[0]/(torch.sum(weight))