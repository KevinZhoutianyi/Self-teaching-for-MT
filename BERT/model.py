from cProfile import label
import os
import random
import numpy as np
import gc
import copy
from transformers import T5ForConditionalGeneration
from transformers import T5Tokenizer
import torch
import torch.nn as nn
import torch.nn.functional as F
from MT_hyperparams import *
import logging


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


class Model(nn.Module):

    def __init__(self, tokenizer, args, name='unknown'):
        super(Model, self).__init__()
        self.name = name
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.vocab_size
        self.name = name
        self.loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
        self.args = args
        self.model = torch.load(
            args.model_name_teacher.replace('/', '')+'.pt').roberta.cuda()
        if(name == 'student' or name == 'student*'):
            self.model = torch.load(
                args.model_name_student.replace('/', '')+'.pt').roberta.cuda()
        self.embedding = Embedding_(self.model.embeddings.word_embeddings).requires_grad_()
        self.linear = nn.Linear(self.model.config.hidden_size,args.out_dim).requires_grad_()

    def forward(self, input_ids, input_attn):

        inp_emb = self.embedding(input_ids)
        last_hidden_state = self.model(inputs_embeds=inp_emb, attention_mask=input_attn).last_hidden_state[:,0,:]
        out = self.linear(last_hidden_state)
        return out

    def get_loss_vec(self, input_ids, input_attn, target):
        logits = self(input_ids,input_attn)
        loss_vec = self.loss_fn(logits,target)
        return logits,loss_vec

    def new(self, name='unknown'):

        model_new = Model( self.tokenizer,
                       args=self.args, name=name).cuda()

        model_new.model.load_state_dict(self.model.state_dict())

        return model_new

