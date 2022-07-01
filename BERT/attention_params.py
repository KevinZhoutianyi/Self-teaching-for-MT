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
class Embedding_(torch.nn.Module):
    def __init__(self, embedding_layer):
        super(Embedding_, self).__init__()
        self.embedding = embedding_layer.cuda()
        # https://github.com/huggingface/transformers/issues/4875

    def forward(self, mask):
        if mask.ndim == 2:
            assert mask.dtype == torch.long
            return self.embedding(mask)

        assert mask.dtype == torch.float
        return torch.matmul(mask, self.embedding.weight)
# class attention_params(torch.nn.Module):# A and B
#     def __init__(self, tokenizer, args):
#         super(attention_params, self).__init__()
        
#         self.act = torch.nn.Sigmoid()
#         self.model = torch.load(
#             args.model_name_teacher.replace('/', '')+'.pt').roberta.cuda()
#         self.embedding = Embedding_(self.model.embeddings.word_embeddings)
#         self.embedding.requires_grad_ = False
#         self.linear = torch.nn.Linear(self.model.config.hidden_size,1).requires_grad_()
#         torch.nn.init.xavier_uniform(self.linear.weight)
        
        
#     def forward(self, x, attn):

        
#         inp_emb = self.embedding(x)
#         last_hidden_state = self.model(inputs_embeds=inp_emb, attention_mask=attn).last_hidden_state[:,0,:]
#         out = self.linear(last_hidden_state)
#         weight = torch.squeeze(self.act(out))
#         weight = torch.clamp(weight, min=0.1,max=0.9)
#         return weight*x.shape[0]/(torch.sum(weight))

class attention_params(torch.nn.Module):
    def __init__(self, N):
        super(attention_params, self).__init__()
        self.alpha = torch.nn.Parameter(torch.ones(N))
        self.softmax = torch.nn.Softmax(dim=-1)
        
    def forward(self, idx):
        
        probs = self.softmax(self.alpha)
        
        return probs[idx]