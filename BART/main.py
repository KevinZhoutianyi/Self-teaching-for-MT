# %%
import torch.nn as nn
import torch
import math
from utils import *
from opt import *
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, SubsetRandomSampler

from datasets import load_dataset,load_metric
import torch
import logging
import numpy as np
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import sys
import time
from transformers.optimization import Adafactor
import os
import gc
from transformers import AutoTokenizer, AutoModel

import torch
import torch.nn.functional as F
import torch.optim as optim


# %%
from transformers import BartForConditionalGeneration, BartConfig
configuration = BartConfig(vocab_size=32768 ,max_position_embeddings=512,d_model = 512,encoder_layers=6,encoder_ffn_dim=2048,encoder_attention_heads=8,\
    dropout=0.1,activation_function='relu')
transformer = BartForConditionalGeneration(configuration).cuda()
transformer.model.encoder.embed_positions = PositionalEncoding(512)

# %%

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
local_test = 0
if(local_test==0):
    max_length= 128
    test_step = 500000
    report_step = 10000
    seed = 2
    bs = 128 
    lr = 1e-4
    train_num = 1000000
    valid_num = 2000
    test_num = 2000
else:
    max_length= 128
    test_step = 1000
    report_step = 100
    seed = 2
    bs = 32
    lr = 1e-4
    train_num = 200
    valid_num = 100
    test_num = 100


test_step = test_step//bs * bs
report_step = report_step//bs * bs

now = time.strftime("%Y-%m-%d-%H_%M_%S",time.localtime(time.time())) 
log_format = '%(asctime)s |\t  %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join("./log/", now+'.txt'),'w',encoding = "UTF-8")
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

logging.info(f"test step: {test_step}")
logging.info(f"rep step: {report_step}")

# Setting the seeds
np.random.seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True
torch.manual_seed(seed)
cudnn.enabled=True
torch.cuda.manual_seed(seed)

# %%

dataset = load_dataset('wmt14','de-en')
train = dataset['train'].shuffle(seed=seed).select(range(train_num))
valid = train[:100]#dataset['validation'].shuffle(seed=seed).select(range(valid_num))
test = dataset['test'].shuffle(seed=seed).select(range(test_num))
train = train['translation']
valid = valid['translation']
test = test['translation']

# %%
from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast

tokenizer = PreTrainedTokenizerFast(tokenizer_file="tokenizer-en-de.json")
tokenizer.add_special_tokens({'pad_token': '[PAD]','unk_token':'[UNK]','eos_token':'[SEP]'})
optimizer = ScheduledOptim(
        optim.Adam(transformer.parameters(), betas=(0.9, 0.98), eps=1e-09),
        1, 512, 4000)
criterion = torch.nn.CrossEntropyLoss(ignore_index = tokenizer.pad_token_id,  label_smoothing=0.1)



# %%



train_data = get_Dataset(train, tokenizer,max_length=max_length)
train_dataloader = DataLoader(train_data, sampler= SequentialSampler(train_data), 
                        batch_size=bs, pin_memory=True, num_workers=4)
valid_data = get_Dataset(valid, tokenizer,max_length=max_length)
valid_dataloader = DataLoader(valid_data, sampler=SequentialSampler(valid_data), 
                        batch_size=bs, pin_memory=True, num_workers=4)
test_data = get_Dataset(test, tokenizer,max_length=max_length)
test_dataloader = DataLoader(test_data, sampler=SequentialSampler(test_data), 
                        batch_size=bs, pin_memory=True, num_workers=4)

# %%
def my_train(_dataloader,model,optimizer,iter):#
    objs = AvgrageMeter()
    
    for step,batch in enumerate(_dataloader):
        iter[0] += batch[0].shape[0]
        optimizer.zero_grad()
        train_x = Variable(batch[0], requires_grad=False).to(device, non_blocking=False)
        train_x_attn = Variable(batch[1], requires_grad=False).to(device, non_blocking=False)
        train_y = Variable(batch[2], requires_grad=False).to(device, non_blocking=False)    
        train_y_attn = Variable(batch[3], requires_grad=False).to(device, non_blocking=False)   
        logits = model(input_ids=train_x, attention_mask=train_x_attn, labels=train_y,decoder_attention_mask= train_y_attn).logits
        # print(logits.shape)
        loss = criterion(logits.view(-1,logits.shape[-1]),train_y.view(-1))
        loss.backward()
        optimizer.step_and_update_lr()
        objs.update(loss.item(), bs)
        if(iter[0]%report_step==0 and iter[0]!=0):
            logging.info(f'iter:{iter[0]}\t,avgloss:{objs.avg}')
            objs.reset()

# %%
import copy
@torch.no_grad()
def my_test(_dataloader,model,epoch):
    # logging.info(f"GPU mem before test:{getGPUMem(device)}%")
    acc = 0
    counter = 0
    model.eval()
    metric_sacrebleu =  load_metric('sacrebleu')
    
    # for step, batch in enumerate(tqdm(_dataloader,desc ="test for epoch"+str(epoch))):
    for step, batch in enumerate(_dataloader):
        
        test_dataloaderx = Variable(batch[0], requires_grad=False).to(device, non_blocking=False)
        test_dataloaderx_attn = Variable(batch[1], requires_grad=False).to(device, non_blocking=False)
        test_dataloadery = Variable(batch[2], requires_grad=False).to(device, non_blocking=False)
        test_dataloadery_attn = Variable(batch[3], requires_grad=False).to(device, non_blocking=False)
        ls = model(input_ids=test_dataloaderx, attention_mask=test_dataloaderx_attn, labels=test_dataloadery, decoder_attention_mask= test_dataloadery_attn).loss
        acc+= ls.item()
        counter+= 1
        pre = model.generate(test_dataloaderx ,num_beams = 4, early_stopping = True, max_length = max_length, length_penalty =0.6, repetition_penalty = 0.8)
        x_decoded = tokenizer.batch_decode(test_dataloaderx,skip_special_tokens=True)
        pred_decoded = tokenizer.batch_decode(pre,skip_special_tokens=True)
        label_decoded =  tokenizer.batch_decode(test_dataloadery,skip_special_tokens=True)
        
        pred_str = [x  for x in pred_decoded]
        label_str = [[x] for x in label_decoded]
        metric_sacrebleu.add_batch(predictions=pred_str, references=label_str)
        if  step%100==0:
            logging.info(f'x_decoded[:2]:{x_decoded[:2]}')
            logging.info(f'pred_decoded[:2]:{pred_decoded[:2]}')
            logging.info(f'label_decoded[:2]:{label_decoded[:2]}')
            
            
    sacrebleu_score = metric_sacrebleu.compute()
    logging.info('sacreBLEU : %f',sacrebleu_score['score'])#TODO:bleu may be wrong cuz max length
    logging.info('test loss : %f',acc/(counter))
    
    
    model.train()
    
    
    # logging.info(f"GPU mem after test:{getGPUMem(device)}%")
        

# %%

# my_test(valid_dataloader,transformer,-1)
for epoch in range(10):
    iter = [0]
    logging.info(f"\n\n  ----------------epoch:{epoch}-----lr:{optimizer._optimizer.param_groups[0]['lr']}-----------")
    my_train(train_dataloader,transformer,optimizer,iter)
    lr = optimizer._optimizer.param_groups[0]['lr']

    my_test(valid_dataloader,transformer,epoch) 
    torch.save(transformer,'./model/'+now+'model.pt')




# %%



