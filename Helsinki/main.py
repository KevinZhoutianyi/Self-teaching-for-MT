
# %%
import os
os.getcwd() 
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import warnings
warnings.filterwarnings("ignore")
from T5 import *
from datasets import load_dataset
from transformers import T5Tokenizer
from MT_hyperparams import *
import torch.backends.cudnn as cudnn
from utils import *
from attention_params import *
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, SubsetRandomSampler
from torch.autograd import Variable
from losses import *
from architect import *
import logging
import sys
import time
from torch.utils.tensorboard import SummaryWriter

# %%
now = time.strftime("%Y-%m-%d-%H_%M_%S",time.localtime(time.time())) 
log_format = '%(asctime)s |\t  %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join("./log/", now+'.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
dataset = load_dataset('opus_euconst','en-fr')
logging.info(dataset)
logging.info(dataset['train'][5])



writer = SummaryWriter('tensorboard')

# Setting the seeds
np.random.seed(seed_)
torch.cuda.set_device(1)
cudnn.benchmark = True
torch.manual_seed(seed_)
cudnn.enabled=True
torch.cuda.manual_seed(seed_)

# %%

pretrained  =  T5ForConditionalGeneration.from_pretrained("t5-base")
torch.save(pretrained,'T5BASE.pt')

# %%
# Load the tokenizer.
import random
tokenizer = T5Tokenizer.from_pretrained("t5-base")

criterion = torch.nn.CrossEntropyLoss( reduction='none')#ignore_index = tokenizer.pad_token_id,
dataset = dataset.shuffle(seed=seed_)
L = len(dataset['train'])
L_t =train_num_points
L_v =valid_num_points
L_test = test_num_points


train = dataset['train']['translation'][:L_t]
valid = dataset['train']['translation'][L_t:L_t+L_v]
test = dataset['train']['translation'][L_t+L_v:L_t+L_v+L_test]
def preprocess(dat):
    for t in dat:
        t['en'] = 'translate English to French:' + t['en']
preprocess(train)
preprocess(valid)
preprocess(test)
logging.info("train len: %d",len(train))
logging.info("valid len: %d",len(valid))
logging.info("test len: %d" ,len(test))
logging.info(train[5])

# %%

train_data = get_train_Dataset(train, tokenizer)# Create the DataLoader for our training set.
train_dataloader = DataLoader(train_data, sampler=SequentialSampler(train_data), 
                        batch_size=batch_size, pin_memory=True, num_workers=0)
valid_data = get_aux_dataset(valid, tokenizer)# Create the DataLoader for our training set.
valid_dataloader = DataLoader(valid_data, sampler=SequentialSampler(valid_data), 
                        batch_size=batch_size, pin_memory=True, num_workers=0)
test_data = get_aux_dataset(test, tokenizer)# Create the DataLoader for our training set.
test_dataloader = DataLoader(test_data, sampler=SequentialSampler(test_data),
                        batch_size=batch_size, pin_memory=True, num_workers=0)#, sampler=RandomSampler(test_data)

# %%

A = attention_params(len(train)//2)#half of train regarded as u
A = A.cuda()

# TODO: model loaded from saved model
model_w = T5(criterion=criterion, tokenizer= tokenizer, name = 'model_w_in_main')
model_w = model_w.cuda()
w_optimizer = torch.optim.SGD(model_w.parameters(),w_lr,momentum=momentum,weight_decay=decay)
scheduler_w  = torch.optim.lr_scheduler.CosineAnnealingLR(w_optimizer, float(epochs), eta_min=learning_rate_min)



model_v = T5(criterion=criterion, tokenizer= tokenizer, name = 'model_v_in_main')
model_v = model_v.cuda()
v_optimizer = torch.optim.SGD(model_v.parameters(),v_lr,momentum=momentum,weight_decay=decay)
scheduler_v  = torch.optim.lr_scheduler.CosineAnnealingLR(v_optimizer, float(epochs), eta_min=learning_rate_min)



architect = Architect(model_w, model_v,  A)

# %%
x = ['my name is kevin','it is my nameit is']
for index,i in enumerate(x) :
    x[index] = 'translate English to French:' + x[index]
y= tokenize(x, tokenizer, max_length = summary_length)
input = y[0].cuda()
output  = model_v.generate(input,max_length=summary_length)
tokenizer.batch_decode(output)

# %%

def my_test(test_dataloader,model,epoch):
    acc = 0
    counter = 0
    model.eval()
    for step, batch in enumerate(test_dataloader):
        test_dataloaderx = Variable(batch[0], requires_grad=False).cuda()
        n = test_dataloaderx.size(0)   
        test_dataloaderx_attn = Variable(batch[1], requires_grad=False).cuda()
        test_dataloadery = Variable(batch[2], requires_grad=False).cuda()
        test_dataloadery_attn = Variable(batch[3], requires_grad=False).cuda()
        # logging.info(f"{model.name}")
        # logging.info(f"test: x {test_dataloaderx}")
        # logging.info(f"test: y {test_dataloadery}")
        # logging.info(f"generate:{model.generate(test_dataloaderx)}")
        ls = my_loss(test_dataloaderx,test_dataloaderx_attn,test_dataloadery,test_dataloadery_attn,model)
        
        # logging.info(f"loss:{ls}")
        acc+= ls
        counter+= 1
    
    logging.info('%s test loss : %f',model.name,acc/(counter*n))
    writer.add_scalar("MT/"+model.name+"test_loss", acc/counter, global_step=epoch)
        

# %%
def my_train(epoch, train_dataloader, valid_dataloader, w_model, v_model, architect, A, w_optimizer, v_optimizer, lr_w, lr_v, ):
    v_trainloss_acc = 0
    w_trainloss_acc = 0
    counter = 0
    for step, batch in enumerate(train_dataloader):
        counter+=1
        # logging.info(" \t\t Step count: %d",step)
        
        batch_loss_w, batch_loss_v,  batch_count = 0, 0, 0
        input_w = Variable(batch[0], requires_grad=False).cuda()
        input_w_attn = Variable(batch[1], requires_grad=False).cuda()
        output_w = Variable(batch[2], requires_grad=False).cuda()
        output_w_attn = Variable(batch[3], requires_grad=False).cuda()        
        input_v = Variable(batch[4], requires_grad=False).cuda()
        input_v_attn = Variable(batch[5], requires_grad=False).cuda()      
        attn_idx = Variable(batch[6], requires_grad=False).cuda()
        
        
        valid_batch = next(iter(valid_dataloader))
        valid_input_v      = Variable(valid_batch[0], requires_grad=False).cuda()
        valid_input_v_attn = Variable(valid_batch[1], requires_grad=False).cuda()
        valid_out_v      = Variable(valid_batch[2], requires_grad=False).cuda()
        valid_out_v_attn = Variable(valid_batch[3], requires_grad=False).cuda()
        

        if epoch <= epochs:
            architect.step(input_w,  output_w,input_w_attn, output_w_attn, w_optimizer, input_v, input_v_attn,valid_input_v, valid_input_v_attn, valid_out_v, 
                valid_out_v_attn, v_optimizer, attn_idx, lr_w, lr_v)

        if epoch <=epochs:
            
            w_optimizer.zero_grad()
            loss_w = CTG_loss(input_w, input_w_attn, output_w, output_w_attn, attn_idx, A, w_model)
            # logging.info(f"loss_w (train):{loss_w}")
            batch_loss_w += loss_w.item()
            loss_w.backward()
            # nn.utils.clip_grad_norm(w_model.parameters(), grad_clip)
            w_optimizer.step()


            v_optimizer.zero_grad()
            loss_aug = calc_loss_aug(input_v, input_v_attn, w_model, v_model)
            v_loss =  (loss_aug)
            # logging.info(f"v_loss (train):{v_loss}")
            batch_loss_v += v_loss.item()
            v_loss.backward()
            # nn.utils.clip_grad_norm(v_model.parameters(), grad_clip)
            v_optimizer.step()     
                
            v_trainloss_acc+=v_loss.item()
            w_trainloss_acc+=loss_w.item()
        if(step*batch_size%5==0):
            logging.info(f"{step*batch_size*100/(train_num_points//2)}%")
    
    logging.info(str(("Attention Weights A : ", A.alpha)))
    return w_trainloss_acc,v_trainloss_acc/counter


# %%

my_test(test_dataloader,model_w,-1) 
my_test(test_dataloader,model_v,-1)  
for epoch in range(epochs):

    lr_w = scheduler_w.get_lr()[0]
    lr_v = scheduler_v.get_lr()[0]

    logging.info(f"\n\n  ----------------epoch:{epoch},\t\tlr_w:{lr_w},\t\tlr_v:{lr_v}----------------")

    w_train_loss,v_train_loss =  my_train(epoch, train_dataloader, valid_dataloader, model_w, model_v,  architect, A, w_optimizer, v_optimizer, lr_w,lr_v)
    
    scheduler_w.step()
    scheduler_v.step()

    writer.add_scalar("MT/w_trainloss", w_train_loss, global_step=epoch)
    writer.add_scalar("MT/v_trainloss", v_train_loss, global_step=epoch)

    logging.info(f"w_train_loss:{w_train_loss},v_train_loss:{v_train_loss}")

    
    my_test(test_dataloader,model_w,epoch) 
    my_test(test_dataloader,model_v,epoch)  

    torch.save(model_v,'model_v.pt')
     
   
   
        
    



# %%
import gc

gc.collect()

torch.cuda.empty_cache()

# %%
def decodizer(x,y,g,loss):
    print("x")
    print(tokenizer.decode(x))
    print("y")
    print(tokenizer.decode(y))
    print("g")
    print(tokenizer.decode(g))
    print(loss)

# %%
decodizer([13959,  1566,    12,  2379,    10,  1570,     8,   262,   188,  3073,
         16494,    63,     8,  1657,     3,   184,  4663,   226,  9457,   117,
          2243,    13,  6923,     3,   184,  4663,   226,  8584,   117,  1522,
            36,  5821,    57,     3,   184,  4663,   226,  9457,   117,  2243,
            13,  6923,    13,     8,  1611,  3545,     3,   184,  4663,   226,
          8584,   117,     3,     5,     1,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0],[ 3039,    90, 10152,   154,   205,  5080,   188,     6,   110,     3,
         25814,     3,   184,  4663,   226,  1206,  5359,   117, 13579,    20,
          4831,     3,   184,  4663,   226,  1206,  7640,   117,   527, 31522,
             7,   260,   110,     3, 25814,     3,   184,  4663,   226,  1206,
          5359,   117, 13579,    20,  4831,    20,     3,    40,    31, 19011,
         15851,     3,   184,  4663,   226,  1206,  7640,   117,     3,     5,
             1,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0],[    0,     0, 32099,     8,     8,     3,     3,     6,     6,     3,
            20,    20,   374,   374,    20,    93,    93,   146,   146,   970,
           970,   146,    20,   247,   247,  3039,  3039,   247,  4692,  4692,
         15648, 15648,  4679,  4679, 15648, 15591, 15591, 14976, 14976, 15591,
          2435,  2435,   597,   597,  2435,   245,   245,    73,    73,   245,
           444,   444,   202,   202,  7443,  7443,  4417,  4417,  7443, 13227,
         13227, 11151, 11151,  3523,  3523, 10711, 10711, 12580, 12580, 16816,
         16816,  5718,  5718,  3570,  3570,  3939,  3939, 14126, 14126,  3939,
          3570,  9026,  9026, 20883, 20883,  6252,  6252, 10562, 10562,     5,
             5,     1],16.631591796875)

# %%
tokenizer.decode( [13959,  1566,    12,  2379,    10,  2962, 18901,     7,    13,     8,
           423,  2243,  1522,    36,  3982,   163,     3,    99, 17310, 12330,
             7,    33,  3823,     5,     1,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0])

# %%
tokenizer.decode([    0,   622, 15172,     7,    20,    50, 13579,  4752,   154, 19134,
             3,    29,    15,   527,     3,  2165,   179,     7,   238,   108,
           285,    29,   776, 21399,     7, 21442,    29,    17,     5,     1])

# %%
tokenizer.decode([    0,   622, 15172,     7,    20,    50, 13579,  4752,   154, 19134,
             3,    29,    15,   527,     3,  2165,   179,     7,   238,   108,
           285,    29,   776, 21399,     7, 21442,    29,    17,     5,     1])

# %%
tokenizer.decode([13959,  1566,    12,  2379,    10,  2266,     1,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0])

# %%
tokenizer.decode([ 4738,   288,  3908,    20,    50,  1419, 23650,    20,    50,  6919,
            20,  2209,  9424,  5672,    20,    50,   377,   154,  3764,  2661,
            20, 30342,   247,    90,  2625,    15,    20,     3,    40,    31,
           154,    40,  8240,  9343,    20,     3,    40,    31, 19011,   117,
             1,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0])

# %%
tokenizer.decode([    0,  7762,    23,  3764,  3569,    50,  1419, 23650,    20,    50,
          6919,    20,  2209,  9424,  5672,   247,    50,   377,   154,  3764,
          2661,    20, 30342,   247,    90,  2625,    15,    20,     3,    40,
            31,   154,    40,  8240,  9343,    20,  7421,    18,    75,    23,
           117,     1])

# %%
tokenizer.decode([0,  6206,  6667,    27,     1])
tokenizer.decode([13959,  1566,    12,  2379,    10, 17608,   994,    27,     1,     0])
logging.info("vocab size : %d",model_v.vocab_size)
logit = torch.load('logits.pt')
target = torch.load('target_ids.pt')
tokenizer.decode(target[0])
logit.shape
_,maxx = torch.max(logit,dim=-1,keepdim=True)
maxx.shape
tokenizer.decode(maxx[0].squeeze(-1))

model_v.embedding

# %%
A.alpha.shape

# %%
tokenizer.decode([13959,  1566,    12,  2379,    10,   634,     3,  2685, 10140,    29,
            30,  5178,    16,     8,  2509,    18,  2206,  8557,    13,     8,
          1611,  1290,  1582,  1208,  2149,     3,  4822,    12,    16,  7491,
          6289,    18, 24151, 14296,   599,    75,    61,    13,     8, 11378,
          1522,  1243,    24,     8,  8541,  1015,  4376,    65, 13841,     8,
          1389, 23460,   257,  6346,     7,   937,    21,    57,     8,  2509,
            18,  2206,  8557,    13,     8,  1611,  1290,  1582,  1208,  2149,
           406,  5274,  7012,     7,    21,    44,   709,     8,   336,   192,
           203,   274,     8,  6498,     5,     1,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0])
tokenizer.decode([    0,   312,     3, 12563,  2970,    20,  5178,   185,     3, 28874,
            20,   483,   146,  4870,  1911,   154,  5785, 18779,  4642,   154,
             3,    85,     3,    40,    31,  8372,  6289,    18, 24151,     6,
          8986,    15,  1914,   500,     3,    75,   201,    20,    50, 11378,
             6, 16558,   238,    90,  3277, 13924,  2410,   154,  1445,    15,
           110,  3157,  2897,  1389,    15,     7,    20, 23460,   257,     3,
         25796,     7,   260,    90, 17100,    20,  1112,   146,  5224,     7,
            17,  5843,  2963,   154,  2046,  6274,  2430,  3890,    35,  1532,
          7012,     7,     3,     7,   154,   208, 12449,  4530,   185,  3000,
           110,  1763,  6062,     7,  5228,  2811,    75,   154,    26,   288,
            90, 10418,     5,     1])



