# %%
import os
os.getcwd() 
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import warnings
warnings.filterwarnings("ignore")
from T5 import *
from datasets import load_dataset,load_metric
from transformers import T5Tokenizer
import torch_optimizer as optim
from transformers.optimization import Adafactor, AdafactorSchedule
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
import transformers
import time
import argparse
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import string

# %%
parser = argparse.ArgumentParser("main")


parser.add_argument('--valid_num_points', type=int,             default = 100, help='validation data number')
parser.add_argument('--train_num_points', type=int,             default = 1000, help='train data number')

parser.add_argument('--batch_size', type=int,                   default=32,     help='Batch size')
parser.add_argument('--train_w_num_points', type=int,           default=8,      help='train_w_num_points for each batch')
parser.add_argument('--train_v_synthetic_num_points', type=int, default=8,      help='train_v_synthetic_num_points for each batch')
parser.add_argument('--train_v_num_points', type=int,           default=8,      help='train_v_num_points for each batch')
parser.add_argument('--train_A_num_points', type=int,           default=8,      help='train_A_num_points decay for each batch')


parser.add_argument('--gpu', type=int,                          default=0,      help='gpu device id')
parser.add_argument('--model_name', type=str,                   default='t5-small',      help='model_name')
parser.add_argument('--exp_name', type=str,                     default='smooth64',      help='experiment name')
parser.add_argument('--rep_num', type=int,                      default='25',      help='howmany step report once')

parser.add_argument('--epochs', type=int,                       default=50,     help='num of training epochs')
parser.add_argument('--pre_epochs', type=int,                   default=0,      help='train model W for x epoch first')
parser.add_argument('--grad_clip', type=float,                  default=1,      help='gradient clipping')
parser.add_argument('--grad_acc_count', type=float,             default=64,      help='gradient accumulate steps')

parser.add_argument('--w_lr', type=float,                       default=1e-3,   help='learning rate for w')
parser.add_argument('--v_lr', type=float,                       default=1e-3,   help='learning rate for v')
parser.add_argument('--A_lr', type=float,                       default=1e-4,   help='learning rate for A')
parser.add_argument('--learning_rate_min', type=float,          default=1e-8,   help='learning_rate_min')
parser.add_argument('--decay', type=float,                      default=1e-3,   help='weight decay')
parser.add_argument('--momentum', type=float,                   default=0.7,    help='momentum')
parser.add_argument('--smoothing', type=float,                   default=0.1,    help='labelsmoothing')


parser.add_argument('--traindata_loss_ratio', type=float,       default=0.9,    help='human translated data ratio')
parser.add_argument('--syndata_loss_ratio', type=float,         default=0.1,    help='augmented dataset ratio')

parser.add_argument('--valid_begin', type=int,                  default=0,      help='whether valid before train')
parser.add_argument('--train_A', type=int,                      default=0 ,     help='whether train A')



args = parser.parse_args()#(args=['--batch_size', '8',  '--no_cuda'])#used in ipynb

# %%
import wandb
os.environ['WANDB_API_KEY']='a166474b1b7ad33a0549adaaec19a2f6d3f91d87'
os.environ['WANDB_NAME']=args.exp_name
# os.environ['WANDB_NOTES']='train without A,withoutAandt5smallandbatch64 '
wandb.init(project="CUDAOOM",config=args)


# %%
now = time.strftime("%Y-%m-%d-%H_%M_%S",time.localtime(time.time())) 

log_format = '%(asctime)s |\t  %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join("./log/", now+'.txt'),'w',encoding = "UTF-8")
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
dataset = load_dataset('wmt14','de-en')

logging.info(args)
logging.info(dataset)
logging.info(dataset['train'][5])



writer = SummaryWriter('tensorboard')

# Setting the seeds
np.random.seed(seed_)
torch.cuda.set_device(args.gpu)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True
torch.manual_seed(seed_)
cudnn.enabled=True
torch.cuda.manual_seed(seed_)

# %%
modelname = args.model_name
pretrained  =  T5ForConditionalGeneration.from_pretrained(modelname)
logging.info(f'modelsize:{count_parameters_in_MB(pretrained)}MB')
torch.save(pretrained,modelname+'.pt')

# %%
# Load the tokenizer.
import random
tokenizer = T5Tokenizer.from_pretrained(modelname)

criterion = torch.nn.CrossEntropyLoss( reduction='none',label_smoothing=0.1)#,ignore_index = tokenizer.pad_token_id)#
# dataset = dataset.shuffle(seed=seed_)
train = dataset['train']['translation'][:args.train_num_points]
valid = dataset['validation']['translation'][:args.valid_num_points]
test = dataset['test']['translation']#[L_t+L_v:L_t+L_v+L_test]
def preprocess(dat):
    for t in dat:
        t['en'] = "translate English to German: " + t['en'] 
preprocess(train)
preprocess(valid)
preprocess(test)
num_batch = args.train_num_points//args.batch_size
train = train[:args.batch_size*num_batch]
logging.info("train len: %d",len(train))
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
logging.info(test[2])

# %%
target_language  = 'de'
train_data = get_train_Dataset(train, tokenizer)# Create the DataLoader for our training set.
logging.info('train data get')
train_dataloader = DataLoader(train_data, sampler=SequentialSampler(train_data), 
                        batch_size=args.batch_size, pin_memory=True, num_workers=4)
logging.info('train data loader get')
valid_data = get_aux_dataset(valid, tokenizer)# Create the DataLoader for our training set.
valid_dataloader = DataLoader(valid_data, sampler=SequentialSampler(valid_data), 
                        batch_size=16, pin_memory=True, num_workers=4)
logging.info('valid data loader get')
test_data = get_aux_dataset(test, tokenizer)# Create the DataLoader for our training set.
test_dataloader = DataLoader(test_data, sampler=SequentialSampler(test_data),
                        batch_size=16, pin_memory=True, num_workers=4)#, sampler=RandomSampler(test_data)
logging.info('test data loader get')

# %%

A = attention_params(train_w_num_points_len)#half of train regarded as u
A = A.cuda()



# TODO: model loaded from saved model
model_w = T5(criterion=criterion, tokenizer= tokenizer, args = args, name = 'model_w_in_main')
model_w = model_w.cuda()
w_optimizer = Adafactor(model_w.parameters(), scale_parameter=True, relative_step=True, warmup_init=False,clip_threshold=1,beta1=0)
scheduler_w  = torch.optim.lr_scheduler.StepLR(w_optimizer,step_size=10, gamma=0.9)



model_v = T5(criterion=criterion, tokenizer= tokenizer, args = args, name = 'model_v_in_main')
model_v = model_v.cuda()
v_optimizer =Adafactor(model_v.parameters(), scale_parameter=True, relative_step=True, warmup_init=False, clip_threshold=1,beta1=0)
scheduler_v  = torch.optim.lr_scheduler.StepLR(v_optimizer,step_size=10, gamma=0.9)



architect = Architect(model_w, model_v,  A, args)

# %%

def my_test(_dataloader,model,epoch):
    acc = 0
    counter = 0
    model.eval()
    metric_sacrebleu =  load_metric('sacrebleu')
    metric_bleu =  load_metric('bleu')

    # for step, batch in enumerate(tqdm(_dataloader,desc ="test for epoch"+str(epoch))):
    for step, batch in enumerate(_dataloader):
        test_dataloaderx = Variable(batch[0], requires_grad=False).to(device, non_blocking=False)
        test_dataloaderx_attn = Variable(batch[1], requires_grad=False).to(device, non_blocking=False)
        test_dataloadery = Variable(batch[2], requires_grad=False).to(device, non_blocking=False)
        test_dataloadery_attn = Variable(batch[3], requires_grad=False).to(device, non_blocking=False)
        with torch.no_grad():
            ls = my_loss(test_dataloaderx,test_dataloaderx_attn,test_dataloadery,test_dataloadery_attn,model)
            acc+= ls
            counter+= 1
            pre = model.generate(test_dataloaderx)
            x_decoded = tokenizer.batch_decode(test_dataloaderx,skip_special_tokens=True)
            pred_decoded = tokenizer.batch_decode(pre,skip_special_tokens=True)
            label_decoded =  tokenizer.batch_decode(test_dataloadery,skip_special_tokens=True)
            
            pred_str = [x.replace('.', '')  for x in pred_decoded]
            label_str = [[x.replace('.', '')] for x in label_decoded]
            pred_list = [x.replace('.', '').split()  for x in pred_decoded]
            label_list = [[x.replace('.', '').split()] for x in label_decoded]
            if  step%100==0:
                logging.info(f'x_decoded[:2]:{x_decoded[:2]}')
                logging.info(f'pred_decoded[:2]:{pred_decoded[:2]}')
                logging.info(f'label_decoded[:2]:{label_decoded[:2]}')
            metric_sacrebleu.add_batch(predictions=pred_str, references=label_str)
            metric_bleu.add_batch(predictions=pred_list, references=label_list)
            
    logging.info('computing score...')            
    sacrebleu_score = metric_sacrebleu.compute()
    bleu_score = metric_bleu.compute()
    logging.info('%s sacreBLEU : %f',model.name,sacrebleu_score['score'])#TODO:bleu may be wrong cuz max length
    logging.info('%s BLEU : %f',model.name,bleu_score['bleu'])
    logging.info('%s test loss : %f',model.name,acc/(counter))
    writer.add_scalar(model.name+"/test_loss", acc/counter, global_step=epoch)
    writer.add_scalar(model.name+"/sacreBLEU",sacrebleu_score['score'], global_step=epoch)
    writer.add_scalar(model.name+"/BLEU",bleu_score['bleu'], global_step=epoch)
    
    wandb.log({'sacreBLEU'+model.name: sacrebleu_score['score']})
    
    wandb.log({'test_loss'+model.name: acc/counter})
    del test_dataloaderx,acc,counter,test_dataloaderx_attn,sacrebleu_score,bleu_score,test_dataloadery,test_dataloadery_attn,ls,pre,x_decoded,pred_decoded,label_decoded,pred_str,label_str,pred_list,label_list
    torch.cuda.empty_cache()
    model.train()
        

# %%
def my_train(epoch, _dataloader, w_model, v_model, architect, A, w_optimizer, v_optimizer, lr_w, lr_v, ):
    objs_w = AvgrageMeter()
    objs_v = AvgrageMeter()
    v_trainloss_acc = 0
    w_trainloss_acc = 0
    wsize = args.train_w_num_points #now  train_x is [num of batch, datasize], so its seperate batch for the code below
    synsize = args.train_v_synthetic_num_points
    vsize = args.train_v_num_points 
    vtrainsize = vsize+synsize
    vtrainsize_total = train_v_num_points_len+train_v_synthetic_num_points_len
    Asize = args.train_A_num_points 
    grad_acc_count = args.grad_acc_count
    loader_len = len(_dataloader)
    split_size=[wsize,synsize,vsize,Asize]
    for step, batch in enumerate(_dataloader) :
        train_x = Variable(batch[0], requires_grad=False).to(device, non_blocking=True)
        train_x_attn = Variable(batch[1], requires_grad=False).to(device, non_blocking=True)
        train_y = Variable(batch[2], requires_grad=False).to(device, non_blocking=True)
        train_y_attn = Variable(batch[3], requires_grad=False).to(device, non_blocking=True) 
        (input_w,input_syn,input_v,input_A_v) = torch.split(train_x,split_size)
        (input_w_attn,input_syn_attn,input_v_attn,input_A_v_attn) = torch.split(train_x_attn,split_size)
        (output_w,_,output_v,output_A_v) = torch.split(train_y,split_size)
        (output_w_attn,_,output_v_attn,output_A_v_attn) = torch.split(train_y_attn,split_size)
        attn_idx = attn_idx_list[wsize*step:(wsize*step+wsize)]
       

        if (epoch <= args.epochs) and (args.train_A == 1) and epoch >= args.pre_epochs:
            architect.step(input_w,  output_w,input_w_attn, output_w_attn, w_optimizer, input_syn, input_syn_attn,input_A_v, input_A_v_attn, output_A_v, 
                output_A_v_attn, v_optimizer, attn_idx, lr_w, lr_v)
        
        
        if  epoch <= args.epochs:
            for p in w_model.parameters():
                p.requires_grad = True
            loss_w = CTG_loss(input_w, input_w_attn, output_w, output_w_attn, attn_idx, A, w_model)
            w_trainloss_acc+=loss_w.item()
            loss_w.backward()
            objs_w.update(loss_w.item(), wsize)
            if ((step + 1) % grad_acc_count == 0) or (step + 1 == loader_len): 
                # nn.utils.clip_grad_norm(w_model.parameters(), args.grad_clip)
                w_optimizer.step()
                w_optimizer.zero_grad()
            for p in w_model.parameters():
                    p.requires_grad = False

        if epoch >= args.pre_epochs and epoch <= args.epochs:
            
            for p in v_model.parameters():
                p.requires_grad = True
            # loss_aug = calc_loss_aug(input_syn, input_syn_attn, w_model, v_model)#,input_v,input_v_attn,output_v,output_v_attn)
            loss = my_loss2(input_v,input_v_attn,output_v,output_v_attn,model_v)
            v_loss =  (args.traindata_loss_ratio*loss)/num_batch
            v_trainloss_acc+=v_loss.item()
            v_loss.backward()
            objs_v.update(v_loss.item(), vtrainsize)
            if ((step + 1) % grad_acc_count == 0) or (step + 1 == loader_len): 
                # nn.utils.clip_grad_norm(v_model.parameters(), args.grad_clip)
                v_optimizer.step()  
                v_optimizer.zero_grad() 
            for p in v_model.parameters():
                    p.requires_grad = False
        
        torch.cuda.empty_cache()
        progress = 100*(step)/(loader_len-1)
        fre = (loader_len//args.rep_num)
        if((step)%fre == 0 or (step)==(loader_len-1)):
            logging.info(f"{progress:5.3}% \t w_loss_avg:{objs_w.avg*train_w_num_points_len:^.7f}\t v_loss_avg:{objs_v.avg*vtrainsize_total:^.7f}")
  
    logging.info(str(("Attention Weights A : ", A.alpha)))
    
    return w_trainloss_acc,v_trainloss_acc


# %%
if(args.valid_begin==1):
    my_test(valid_dataloader,model_w,-1) #before train
    # my_test(valid_dataloader,model_v,-1)  
for epoch in range(args.epochs):
    lr_w = scheduler_w.get_lr()[0]
    lr_v = scheduler_v.get_lr()[0]

    logging.info(f"\n\n  ----------------epoch:{epoch},\t\tlr_w:{lr_w},\t\tlr_v:{lr_v}----------------")

    w_train_loss,v_train_loss =  my_train(epoch, train_dataloader, model_w, model_v,  architect, A, w_optimizer, v_optimizer, lr_w,lr_v)
    
    scheduler_w.step()
    scheduler_v.step()

    writer.add_scalar("MT/model_w_in_main/w_trainloss", w_train_loss, global_step=epoch)
    writer.add_scalar("MT/model_v_in_main/v_trainloss", v_train_loss, global_step=epoch)

    logging.info(f"w_train_loss:{w_train_loss},v_train_loss:{v_train_loss}")
    wandb.log({'w_train_loss': w_train_loss, 'v_train_loss':v_train_loss})

    
    my_test(valid_dataloader,model_w,epoch) 
    my_test(valid_dataloader,model_v,epoch)  

torch.save(model_v,'./model/'+now+'model_w.pt')
torch.save(model_v,'./model/'+now+'model_v.pt')




