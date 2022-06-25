# %%
import os
os.getcwd() 
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import warnings
warnings.filterwarnings("ignore")
from T5 import *
import torch
from datasets import load_dataset,load_metric
from transformers import T5Tokenizer,AutoModelForSeq2SeqLM, AutoTokenizer
import torch_optimizer as optim
from transformers.optimization import Adafactor, AdafactorSchedule
from MT_hyperparams import seed_,max_length,target_language
import torch.backends.cudnn as cudnn
from utils import *
from attention_params import *
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, SubsetRandomSampler
from torch.autograd import Variable
from losses import *
from architect import *
import logging
import sys
import transformers
import time
import argparse
from tqdm import tqdm
import string
from torch.optim.lr_scheduler import LambdaLR
from os.path import exists
from torch.optim.lr_scheduler import StepLR

# %%
parser = argparse.ArgumentParser("main")


parser.add_argument('--valid_num_points', type=int,             default = 10, help='validation data number')
parser.add_argument('--train_num_points', type=int,             default = 2000, help='train data number')
parser.add_argument('--test_num_points', type=int,              default = 50, help='train data number')

parser.add_argument('--batch_size', type=int,                   default=16,     help='Batch size')
parser.add_argument('--train_w_num_points', type=int,           default=4,      help='train_w_num_points for each batch')
parser.add_argument('--train_v_synthetic_num_points', type=int, default=8,      help='train_v_synthetic_num_points for each batch')
parser.add_argument('--train_v_num_points', type=int,           default=0,      help='train_v_num_points for each batch')
parser.add_argument('--train_A_num_points', type=int,           default=4,      help='train_A_num_points decay for each batch')

parser.add_argument('--gpu', type=int,                          default=0,      help='gpu device id')
parser.add_argument('--num_workers', type=int,                  default=0,      help='num_workers')
parser.add_argument('--model_name_teacher', type=str,           default='google/t5-small-lm-adapt',      help='model_name')
parser.add_argument('--model_name_student', type=str,           default='google/t5-small-lm-adapt',      help='model_name')
parser.add_argument('--model_name_de2en', type=str,             default='Onlydrinkwater/t5-small-de-en-mt',      help='model_name')
parser.add_argument('--exp_name', type=str,                     default='T5spec',      help='experiment name')
parser.add_argument('--rep_num', type=int,                      default=50,      help='report times for 1 epoch')
parser.add_argument('--test_num', type=int,                     default=8000,      help='test times for 1 epoch')

parser.add_argument('--epochs', type=int,                       default=500,     help='num of training epochs')
parser.add_argument('--pre_epochs', type=int,                   default=0,      help='train model W for x epoch first')
parser.add_argument('--grad_clip', type=float,                  default=1,      help='gradient clipping')
parser.add_argument('--grad_acc_count', type=float,             default=-1,      help='gradient accumulate steps')

parser.add_argument('--w_lr', type=float,                       default=1e-3,   help='learning rate for w')
parser.add_argument('--unrolled_w_lr', type=float,              default=1e-3,   help='learning rate for w')
parser.add_argument('--v_lr', type=float,                       default=1e-3,   help='learning rate for v')
parser.add_argument('--unrolled_v_lr', type=float,              default=1e-3,   help='learning rate for v')
parser.add_argument('--A_lr', type=float,                       default=1e-3,   help='learning rate for A')
parser.add_argument('--learning_rate_min', type=float,          default=1e-8,   help='learning_rate_min')
parser.add_argument('--decay', type=float,                      default=1e-3,   help='weight decay')
parser.add_argument('--beta1', type=float,                      default=0,    help='momentum')
parser.add_argument('--beta2', type=float,                      default=0,    help='momentum')
parser.add_argument('--warm', type=float,                       default=10,    help='warmup step')
parser.add_argument('--num_step_lr', type=float,                default=2,    help='warmup step')
parser.add_argument('--decay_lr', type=float,                   default=0.7,    help='warmup step')
# parser.add_argument('--smoothing', type=float,                  default=0.1,    help='labelsmoothing')

parser.add_argument('--freeze', type=int,                       default=1,    help='whether freeze the pretrained encoder')

parser.add_argument('--traindata_loss_ratio', type=float,       default=0,    help='human translated data ratio')
parser.add_argument('--syndata_loss_ratio', type=float,         default=1,    help='augmented dataset ratio')

parser.add_argument('--valid_begin', type=int,                  default=1,      help='whether valid before train')
parser.add_argument('--train_A', type=int,                      default=1 ,     help='whether train A')



args = parser.parse_args()#(args=['--batch_size', '8',  '--no_cuda'])#used in ipynb
args.test_num = args.test_num//args.batch_size * args.batch_size
args.rep_num = args.rep_num//args.batch_size * args.batch_size

# %%
#https://wandb.ai/ check the running status online
import wandb
os.environ['WANDB_API_KEY']='a166474b1b7ad33a0549adaaec19a2f6d3f91d87'
os.environ['WANDB_NAME']=args.exp_name
os.environ["TOKENIZERS_PARALLELISM"] = "false"
wandb.init(project="Selftraining",config=args)


# %%
# logging file
now = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))

log_format = '%(asctime)s |\t  %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(
    "./log/", now+'.txt'), 'w', encoding="UTF-8")
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
dataset = load_dataset('wmt14', 'de-en')

logging.info(args)
logging.info(dataset)
logging.info(dataset['train'][5])


# Setting the seeds
np.random.seed(seed_)
torch.cuda.set_device(args.gpu)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True
torch.manual_seed(seed_)
cudnn.enabled = True
torch.cuda.manual_seed(seed_)


# %%
modelname = args.model_name_teacher
pretrained  =  AutoModelForSeq2SeqLM.from_pretrained(modelname)
pathname = modelname.replace('/','')
logging.info(f'modelsize:{count_parameters_in_MB(pretrained)}MB')

if(exists(pathname+'.pt')==False):
    logging.info(f'saving to {pathname}')
    torch.save(pretrained,pathname+'.pt')

modelname = args.model_name_student
pretrained  =  AutoModelForSeq2SeqLM.from_pretrained(modelname)
pathname = modelname.replace('/','')
logging.info(f'modelsize:{count_parameters_in_MB(pretrained)}MB')
if(exists(pathname+'.pt')==False):
    logging.info(f'saving to {pathname}')
    torch.save(pretrained,pathname+'.pt')

modelname = args.model_name_de2en
pretrained  =  AutoModelForSeq2SeqLM.from_pretrained(modelname)
pathname = modelname.replace('/','')
logging.info(f'modelsize:{count_parameters_in_MB(pretrained)}MB')
if(exists(pathname+'.pt')==False):
    logging.info(f'saving to {pathname}')
    torch.save(pretrained,pathname+'.pt')

  

# %%

# preprocess the data, make a dataloader
import random
modelname = args.model_name_teacher
tokenizer = AutoTokenizer.from_pretrained('t5-small')
criterion = torch.nn.CrossEntropyLoss( reduction='none')#teacher shouldn't have label smoothing, especially when student got same size.
criterion_v = torch.nn.CrossEntropyLoss( reduction='none')#,label_smoothing=args.smoothing) #without LS, V may be too confident to that syn data, and LS do well for real data also.



train = dataset['train'].shuffle(seed=seed_).select(range(args.train_num_points))
valid = dataset['validation'].shuffle(seed=seed_).select(range(args.valid_num_points))
test = dataset['test'].shuffle(seed=seed_).select(range(args.test_num_points))#[L_t+L_v:L_t+L_v+L_test]
train = train['translation']
valid = valid['translation']
test = test['translation']
def preprocess(dat):
    for t in dat:
        t['en'] = "translate English to German: " + t['en']  #needed for T5
preprocess(train)
preprocess(valid)
preprocess(test)
#TODO: Syn_input should be monolingual data, should try en-fo's en. cuz wmt may align
num_batch = args.train_num_points//args.batch_size
train = train[:args.batch_size*num_batch]
logging.info("train len: %d",len(train))

'''
each mini batch consist of : 
1. data to train W
2. monolingual data to generate parallel data
3. data to train V
4. data to train A
'''
train_w_num_points_len = num_batch * args.train_w_num_points
train_v_synthetic_num_points_len = num_batch * args.train_v_synthetic_num_points
train_v_num_points_len = num_batch * args.train_v_num_points
train_A_num_points_len = num_batch * args.train_A_num_points
logging.info("train_w_num_points_len: %d",train_w_num_points_len)
logging.info("train_v_synthetic_num_points_len: %d",train_v_synthetic_num_points_len)
logging.info("train_v_num_points_len: %d",train_v_num_points_len)
logging.info("train_A_num_points_len: %d",train_A_num_points_len)

attn_idx_list = torch.arange(train_w_num_points_len).cuda()
logging.info("valid len: %d",len(valid))
logging.info("test len: %d" ,len(test))
logging.info(train[2])
logging.info(valid[2])
# logging.info(test[2])

# %%
target_language  = 'de'
train_data = get_train_Dataset(train, tokenizer)# Create the DataLoader for our training set.
logging.info('train data get')
train_dataloader = DataLoader(train_data, sampler= SequentialSampler(train_data), 
                        batch_size=args.batch_size, pin_memory=args.num_workers>0, num_workers=args.num_workers)
logging.info('train data loader get')
valid_data = get_aux_dataset(valid, tokenizer)# Create the DataLoader for our training set.
valid_dataloader = DataLoader(valid_data, sampler=SequentialSampler(valid_data), 
                        batch_size=args.batch_size, pin_memory=args.num_workers>0, num_workers=args.num_workers)
logging.info('valid data loader get')
test_data = get_aux_dataset(test, tokenizer)# Create the DataLoader for our training set.
test_dataloader = DataLoader(test_data, sampler=SequentialSampler(test_data),
                        batch_size=args.batch_size, pin_memory=args.num_workers>0, num_workers=args.num_workers)#, sampler=RandomSampler(test_data)
logging.info('test data loader get')

# %%

A = attention_params(args)#half of train regarded as u
A = A.cuda()
# B = load()[0]
# A.load_state_dict(B)



# TODO: model loaded from saved model
model_w = T5(criterion=criterion, tokenizer= tokenizer, args = args, name = 'model_w_in_main')
model_w = model_w.cuda()
w_optimizer = torch.optim.Adam(model_w.parameters(),  lr= args.w_lr ,  betas=(args.beta1, args.beta2) ,eps=1e-9 )
# w_optimizer = Adafactor(model_w.parameters(), lr = args.w_lr ,scale_parameter=False, relative_step=False , warmup_init=False,clip_threshold=1,beta1=0,eps=( 1e-30,0.001))
scheduler_w  =   StepLR(w_optimizer, step_size=args.num_step_lr, gamma=args.decay_lr)
# scheduler_w  = Scheduler(w_optimizer,dim_embed=512, warmup_steps=args.warm, initlr = args.w_lr)



model_v = T5(criterion=criterion_v, tokenizer= tokenizer, args = args, name = 'model_v_in_main')
model_v = model_v.cuda()
v_optimizer = torch.optim.Adam(model_v.parameters(),  lr= args.v_lr ,  betas=(args.beta1,args.beta2) ,eps=1e-9  )
# v_optimizer =Adafactor(model_v.parameters(), lr = args.v_lr ,scale_parameter=False, relative_step=False , warmup_init=False,clip_threshold=1,beta1=0,eps=( 1e-30,0.001))
scheduler_v  =   StepLR(v_optimizer, step_size=args.num_step_lr, gamma=args.decay_lr)
# scheduler_v  = Scheduler(v_optimizer,dim_embed=512, warmup_steps=args.warm, initlr = args.v_lr)


architect = Architect(model_w, model_v,  A, args)


# %%
@torch.no_grad()
def my_test(_dataloader,model,epoch):
    # logging.info(f"GPU mem before test:{getGPUMem(device)}%")
    acc = 0
    counter = 0
    model.eval()
    metric_sacrebleu =  load_metric('sacrebleu')
    # metric_bleu =  load_metric('bleu')
    
    for step, batch in enumerate(_dataloader):
        
        test_dataloaderx = Variable(batch[0], requires_grad=False).to(device, non_blocking=False)[:args.train_w_num_points]
        test_dataloaderx_attn = Variable(batch[1], requires_grad=False).to(device, non_blocking=False)[:args.train_w_num_points]
        test_dataloadery = Variable(batch[2], requires_grad=False).to(device, non_blocking=False)[:args.train_w_num_points]
        test_dataloadery_attn = Variable(batch[3], requires_grad=False).to(device, non_blocking=False)[:args.train_w_num_points]
        ls = my_loss(test_dataloaderx,test_dataloaderx_attn,test_dataloadery,test_dataloadery_attn,model)
        acc+= ls.item()
        counter+= 1
        pre = model.generate(test_dataloaderx)
        x_decoded = tokenizer.batch_decode(test_dataloaderx,skip_special_tokens=True)
        pred_decoded = tokenizer.batch_decode(pre,skip_special_tokens=True)
        label_decoded =  tokenizer.batch_decode(test_dataloadery,skip_special_tokens=True)
        
        pred_str = [x  for x in pred_decoded]
        label_str = [[x] for x in label_decoded]
        # pred_list = [x.split()  for x in pred_decoded]
        # label_list = [[x.split()] for x in label_decoded]
        metric_sacrebleu.add_batch(predictions=pred_str, references=label_str)
        # metric_bleu.add_batch(predictions=pred_list, references=label_list)
        if  step==0:
            logging.info(f'x_decoded[:2]:{x_decoded[:2]}')
            logging.info(f'pred_decoded[:2]:{pred_decoded[:2]}')
            logging.info(f'label_decoded[:2]:{label_decoded[:2]}')
            
            
    logging.info('computing score...') 
    sacrebleu_score = metric_sacrebleu.compute()
    # bleu_score = metric_bleu.compute()
    logging.info('%s sacreBLEU : %f',model.name,sacrebleu_score['score'])#TODO:bleu may be wrong cuz max length
    # logging.info('%s BLEU : %f',model.name,bleu_score['bleu'])
    logging.info('%s test loss : %f',model.name,acc/(counter))
    wandb.log({'sacreBLEU'+model.name: sacrebleu_score['score']})
    wandb.log({'test_loss'+model.name: acc/counter})
    # del test_dataloaderx,acc,counter,test_dataloaderx_attn,sacrebleu_score,bleu_score,test_dataloadery,test_dataloadery_attn,ls,pre,x_decoded,pred_decoded,label_decoded,pred_str,label_str,pred_list,label_list
    # gc.collect()
    # torch.cuda.empty_cache()
    model.eval()

        

# %%
def my_train(epoch, _dataloader, validdataloader, w_model, v_model, architect, A, w_optimizer, v_optimizer, lr_w, lr_v, tot_iter):

    objs_w = AvgrageMeter()
    objs_v_syn = AvgrageMeter()
    objs_v_train = AvgrageMeter()
    objs_v_star_val = AvgrageMeter()
    objs_v_val = AvgrageMeter()
    improvement = 0
    w_trainloss_acc = 0
    # now  train_x is [num of batch, datasize], so its seperate batch for the code below
    wsize = args.train_w_num_points
    synsize = args.train_v_synthetic_num_points
    vsize = args.train_v_num_points
    Asize = args.train_A_num_points
    loader_len = len(_dataloader)
    split_size = [wsize, synsize, vsize, Asize]
    bs = args.batch_size
    w_model.eval()
    v_model.eval()

    # logging.info(f"split size:{split_size}")
    for step, batch in enumerate(_dataloader):
        tot_iter[0] += bs
        

        # logging.info(f"GPU mem :{getGPUMem(device)}%")
        train_x = Variable(batch[0], requires_grad=False).to(
            device, non_blocking=False)
        train_x_attn = Variable(batch[1], requires_grad=False).to(
            device, non_blocking=False)
        train_y = Variable(batch[2], requires_grad=False).to(
            device, non_blocking=False)
        train_y_attn = Variable(batch[3], requires_grad=False).to(
            device, non_blocking=False)
        (input_w, input_syn, input_v, input_A_v) = torch.split(train_x, split_size)
        (input_w_attn, input_syn_attn, input_v_attn,
         input_A_v_attn) = torch.split(train_x_attn, split_size)
        (output_w, _, output_v, output_A_v) = torch.split(train_y, split_size)
        (output_w_attn, _, output_v_attn, output_A_v_attn) = torch.split(
            train_y_attn, split_size)
            
        if(True):# let v train on syn data and w data
            input_v = input_w
            input_v_attn = input_w_attn
            output_v = output_w
            output_v_attn = output_w_attn
            vsize = wsize


        # output_w[step%wsize]+=1 # noise input
        if (args.train_A == 1 and epoch>=args.pre_epochs):
            epsilon_w = args.unrolled_w_lr
            epsilon_v  = args.unrolled_v_lr
            v_star_val_loss = architect.step(input_w,  output_w, input_w_attn, output_w_attn, w_optimizer,
                                             input_v, input_v_attn, output_v, output_v_attn, input_syn, input_syn_attn,
                                             input_A_v, input_A_v_attn, output_A_v, output_A_v_attn, v_optimizer,
                                             epsilon_w, epsilon_v)
            objs_v_star_val.update(v_star_val_loss, Asize)
                            



        w_optimizer.zero_grad()
        loss_w = CTG_loss(input_w, input_w_attn, output_w,
                          output_w_attn, A, w_model)
        w_trainloss_acc += loss_w.item()
        loss_w.backward()
        objs_w.update(loss_w.item(), wsize)
        w_optimizer.step()


        if(epoch>=args.pre_epochs):  
            v_optimizer.zero_grad()
            loss_aug = calc_loss_aug(input_syn, input_syn_attn, w_model, v_model)
            loss = my_loss2(input_v, input_v_attn, output_v,
                            output_v_attn, v_model)
            v_loss = (args.traindata_loss_ratio*loss +
                    loss_aug*args.syndata_loss_ratio)
            v_loss.backward()
            objs_v_syn.update(loss_aug.item(), synsize)
            objs_v_train.update(loss.item(), vsize)
            v_optimizer.step()
            with torch.no_grad():
                valloss = my_loss2(input_A_v, input_A_v_attn,  output_A_v, output_A_v_attn,v_model)
                objs_v_val.update(valloss.item(), Asize)
                improvement += (v_star_val_loss-valloss.item())
                
        progress = 100*(step)/(loader_len-1)
        if(tot_iter[0] % args.test_num == 0 and tot_iter[0] != 0):
            my_test(validdataloader, model_w, epoch)
            my_test(validdataloader, model_v, epoch)
            # logging.info(str(("Attention Weights A : ", A.ReLU(A.alpha))))
            torch.save(model_w,'./model/'+'model_w.pt')#+now+
            torch.save(model_v,'./model/'+'model_v.pt')
            torch.save(A,'./model/'+'A.pt')
            torch.save(model_w.state_dict(),os.path.join(wandb.run.dir, "model_w.pt"))
            torch.save(model_v.state_dict(),os.path.join(wandb.run.dir, "model_v.pt"))
            torch.save(A.state_dict(),os.path.join(wandb.run.dir, "A.pt"))
            wandb.save("./files/*.pt", base_path="./files", policy="live")

        if(tot_iter[0] % args.rep_num == 0 and tot_iter[0] != 0):
            logging.info(f"{progress:5.3}%:\t  W_train_loss:{objs_w.avg:^.7f}\tV_train_syn_loss:{objs_v_syn.avg:^.7f}\tV_train_loss:{objs_v_train.avg:^.7f}\t  V_star_val_loss:{objs_v_star_val.avg:^.7f}\t  improvement:{(objs_v_star_val.avg-objs_v_val.avg):^.7f}")
            with torch.no_grad():
                temp = A(input_w, input_w_attn, output_w,output_w_attn)
            logging.info(f"weight:{temp}")
            # logging.info(f'noise input weight:{temp[step%wsize]}')
            wandb.log({'W_train_loss': objs_w.avg})
            wandb.log({'V_train_syn_loss': objs_v_syn.avg})
            wandb.log({'V_train_loss': objs_v_train.avg})
            wandb.log({'V_star_val_loss': objs_v_star_val.avg})
            wandb.log({'V_val_loss': objs_v_val.avg})
            objs_v_syn.reset()
            objs_v_train.reset()
            objs_w.reset()
            objs_v_star_val.reset()
            objs_v_val.reset()
    return w_trainloss_acc, improvement


# %%
# if(args.valid_begin==1):
#     my_test(valid_dataloader,model_w,-1) #before train
#     my_test(valid_dataloader,model_v,-1)  

tot_iter = [0]
for epoch in range(args.epochs):
    lr_w = scheduler_w.get_lr()[0]
    lr_v = scheduler_v.get_lr()[0]
    lr_A = architect.scheduler_A.get_lr()[0]

    logging.info(f"\n\n  ----------------epoch:{epoch},\t\tlr_w:{lr_w},\t\tlr_v:{lr_v},\t\tlr_A:{lr_A}----------------")

    w_train_loss,improvement =  my_train(epoch, train_dataloader, valid_dataloader, model_w, model_v,  architect, A, w_optimizer, v_optimizer, lr_w,lr_v,tot_iter)
    
    scheduler_w.step()
    scheduler_v.step()
    architect.scheduler_A.step()


    logging.info(f"w_train_loss:{w_train_loss},improvement:{improvement}")
    # wandb.log({'w_train_loss': w_train_loss, 'v_train_loss':v_train_loss})



torch.save(model_v,'./model/'+now+'model_w.pt')
torch.save(model_v,'./model/'+now+'model_v.pt')



# %%


# %%
for (k,v),b in zip(A.named_parameters(),B.values()):
    print(k)
    print(torch.sum(abs(b-v)))

# %%
for x in B.values():
    print(x)

# %%



