import os
import random
import torch
import numpy as np
from MT_hyperparams import *
from utils import *
from model import ClassifierModel

def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
seed_torch(seed_)

class attention_params(torch.nn.Module):# A and B
    def __init__(self, vocab, args):
        super(attention_params, self).__init__()
        self.model = ClassifierModel(vocab,args,'A').cuda()
        self.Sigmoid = torch.nn.Sigmoid()
        
        
    def forward(self, x):
        weight = self.model(x)
        weight = self.Sigmoid(torch.sum(weight,-1))
        weight = torch.clamp(weight, min=0.1,max=0.9)
        return weight*x.shape[0]/(torch.sum(weight))