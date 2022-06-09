import os
import random
import torch
import numpy as np
from MT_hyperparams import *

def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
seed_torch(seed_)

class attention_params(torch.nn.Module):# A and B
    def __init__(self, N):
        super(attention_params, self).__init__()
        self.alpha = torch.nn.Parameter(torch.ones(N))
        self.sigmoid = torch.nn.Sigmoid()
        
    def forward(self, idx):
        probs = self.sigmoid(self.alpha[idx])
        
        return probs