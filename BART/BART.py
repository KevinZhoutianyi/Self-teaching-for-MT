import os
import random
import numpy as np
import copy
from transformers import  T5ForConditionalGeneration
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
        #https://github.com/huggingface/transformers/issues/4875
    def forward(self, mask):
        if mask.ndim == 2:
            assert mask.dtype == torch.long
            return self.embedding(mask)
        
        assert mask.dtype == torch.float
        return torch.matmul(mask, self.embedding.weight[:32100,:])


class BART(nn.Module):
    
    def __init__(self, criterion, tokenizer,name='unknown', MODEL = 't5-base'):
        super(BART, self).__init__()
        self.name = name
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.vocab_size
        self.pad_token_id = tokenizer.pad_token_id
        
        self._criterion = criterion

        self.model = torch.load("pretrained_model.pt")
        
        # # print('Loading the pretrained model ....')
        # Load the pre-trained model trained for 
        # self.model.load_state_dict(torch.load('pretrained_BART.pt'))
        # # print('Done! Loaded the pretrained model ....')
        
        self.encoder = self.model.get_encoder()
        # self.decoder = self.model.get_decoder()
        # print(self.encoder.embed_tokens.weight)
        # print(self.decoder.embed_tokens.weight)

        # embedding layer for both encoder and decoder since it is shared   
        self.embedding = Embedding_(self.encoder.embed_tokens).requires_grad_()#convert token to 512dimensions vector
        self.enc_emb_scale = 1

    def forward(self, input_ids, input_attn, target_ids = None, target_attn = None):
        # print('start of forward')
        # logging.info(f"[T5] input_ids shape {input_ids.shape}")
        
        inp_emb = self.embedding(input_ids)/self.enc_emb_scale
        # logging.info(f"[T5] inp_emb shape {inp_emb.shape}")
        # logging.info(f"[T5] input_attn shape {input_attn.shape}")
        # print("T5 inputshape:",inp_emb.shape,input_attn.shape) # after embedding the shape becomes([5, 232, 768]) (batchsize,tokenziedlength,embeddinglength)
        out = self.model(inputs_embeds = inp_emb, attention_mask = input_attn, labels = target_ids, decoder_attention_mask = target_attn, return_dict=True)

        # print('end of forward')
        return out
    
    def loss(self, input_ids, input_attn, target_ids, target_attn):
        # input is distribution , output is just category index
        
        # print(input_ids.shape)
        batch_size = target_ids.shape[0]
        out_emb = self.embedding(target_ids)/self.enc_emb_scale
        inp_emb = self.embedding(input_ids)/self.enc_emb_scale

        # print(out_emb.shape)
        # print(inp_emb.shape)
        #  is embedding sharing for input and out? cuz they are in different language:  yes
        logits = self.model(inputs_embeds = inp_emb, attention_mask = input_attn, decoder_inputs_embeds   = out_emb, decoder_attention_mask = target_attn, return_dict=True).logits

     
        logits = logits[:,:,:32100]
       
        ## TODO: now we dont use ignoreindex
        loss = self._criterion(logits.view(-1, logits.size(-1)),target_ids.view(-1, target_ids.size(-1)))
        
        loss = loss.view(batch_size, -1).mean(dim = 1)
        loss = torch.mean(loss)
        return loss


 # input is category index , output is onehot
        # output = self.generate(input_ids)
        # logits = (self(input_ids, input_attn, target_ids = output, target_attn = torch.ones_like(output))).logits
      
        # logits = logits[ :,:,:32100]
        
        # print("logits",logits.shape,"tar",target_ids.shape)
        # loss = self._criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1, target_ids.size(-1)))
        # # print("logits",logits.view(-1, logits.size(-1)).shape,"tar",target_ids.view(-1).shape)
        # # print("T5 loss", loss.shape)
        # return loss
    def get_loss_vec(self, input_ids, input_attn, target_ids = None, target_attn = None):

        # batch size
        # print("start of : get_loss_vec")
        batch_size = target_ids.shape[0]
        
        # target_sequence_length of the model
        target_sequence_length = target_ids.shape[1]
        # # print("getlossvec,loss input",input_ids.shape,target_ids.shape)
        logits = (self(input_ids, input_attn, target_ids = target_ids, target_attn = target_attn)).logits
        # # print("17.17",logits.shape,target_ids.shape) #17.17 torch.Size([2, 100, 32128]) torch.Size([2, 100])
        
        

        # torch.save(logits,"./logits.pt")
        # torch.save(target_ids,"./target_ids.pt")
        loss_vec = self._criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))
        # # print("getlossvec,loss_vec",loss_vec.shape)
        loss_vec = loss_vec.view(batch_size, -1).mean(dim = 1)

        # # print("getlossvec,loss_vec",loss_vec.shape)
        # print("end of : get_loss_vec")
        return loss_vec

    # used for generation of summaries from articles
    def generate(self, input_ids, num_beams = 4, max_length=max_length):
        
        # beam search
        # print("start of : generate")
        
        output_ids = self.model.generate( input_ids = input_ids, num_beams = num_beams, early_stopping = True, max_length = max_length, no_repeat_ngram_size = 2, min_length=0,repetition_penalty = 0.8)
        
        
        ## sampling with top_p
        # output_ids = self.model.generate( input_ids = input_ids, num_beams = 1, max_length = max_length, top_p = 0.95, top_k = 50, no_repeat_ngram_size = 2, repetition_penalty = 1.2)

        # print("end of : generate")
        return output_ids

    # new model for the definitions of gradients in architec.py 
    def new(self):

        # there is embedding layer and the summarization head that we will not train on 
        # we just train on the encoder and the decoder weights 
        model_new = T5(self._criterion, self.tokenizer).cuda()
        
        # hence we deep copy all the weights and update the required ones
        # use the first order approximation for the summarization head
        # i.e, do not use unrolled model w.r.t to the summarization head parameters
        model_new.model.load_state_dict(self.model.state_dict())
        
        return model_new


if __name__ == "__main__":
    # print("T5 main")
    t5_tokenizer = T5Tokenizer.from_pretrained('t5-base')
    t5_criterion = torch.nn.CrossEntropyLoss(ignore_index = T5Tokenizer.pad_token_id, reduction='none')
    t5_criterion = t5_criterion.cuda()
    t5 = T5(t5_criterion,t5_tokenizer)
    t5 = t5.cuda()
    # print(t5)

    

    t5new  = t5.new()
    # print(t5new)