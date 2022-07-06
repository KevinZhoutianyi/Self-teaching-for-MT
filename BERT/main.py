# %%
import os
os.getcwd() 
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import warnings
warnings.filterwarnings("ignore")
from model import *
import torch
from datasets import load_dataset,load_metric
from transformers import  AutoTokenizer
import torch_optimizer as optim
from transformers.optimization import Adafactor, AdafactorSchedule
from MT_hyperparams import seed_,max_length
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
from transformers import get_linear_schedule_with_warmup

# %%
parser = argparse.ArgumentParser("main")


parser.add_argument('--valid_num_points', type=int,             default = -1, help='validation data number')
parser.add_argument('--train_w_num_points', type=int,           default = 4000, help='train data number')
parser.add_argument('--train_A_num_points', type=int,           default = 2000, help='train data number')
parser.add_argument('--unlabel_num_points', type=int,           default = 2000, help='train data number')
parser.add_argument('--test_num_points', type=int,              default = -1, help='train data number')

parser.add_argument('--batch_size', type=int,                   default=32,     help='Batch size for test and validation')

parser.add_argument('--w_bs', type=int,                         default=16,      help='train_w_num_points for each batch')
parser.add_argument('--syn_bs', type=int,                       default=8,      help='train_v_synthetic_num_points for each batch')
# parser.add_argument('--train_v_num_points', type=int,           default=0,      help='train_v_num_points for each batch')
parser.add_argument('--A_bs', type=int,                         default=8,      help='train_A_num_points decay for each batch')

parser.add_argument('--gpu', type=int,                          default=0,      help='gpu device id')
parser.add_argument('--num_workers', type=int,                  default=0,      help='num_workers')
parser.add_argument('--model_name_teacher', type=str,           default='roberta-base',      help='model_name')
parser.add_argument('--model_name_student', type=str,           default='roberta-base',      help='model_name')
parser.add_argument('--model_name_de2en', type=str,             default='roberta-base',      help='model_name')
parser.add_argument('--exp_name', type=str,                     default='yelp',      help='experiment name')
parser.add_argument('--rep_num', type=int,                      default=-1,      help='report times for 1 epoch')
parser.add_argument('--test_num', type=int,                     default=-1,      help='test times for 1 epoch')

parser.add_argument('--epochs', type=int,                       default=10,     help='num of training epochs')
parser.add_argument('--pre_epochs', type=int,                   default=0,      help='train model W for x epoch first')
parser.add_argument('--grad_clip', type=float,                  default=1,      help='gradient clipping')
# parser.add_argument('--grad_acc_count', type=float,             default=-1,      help='gradient accumulate steps')

parser.add_argument('--w_lr', type=float,                       default=2e-6,   help='learning rate for w')
parser.add_argument('--unrolled_w_lr', type=float,              default=2e-6,   help='learning rate for w')
parser.add_argument('--v_lr', type=float,                       default=2e-6,   help='learning rate for v')
parser.add_argument('--unrolled_v_lr', type=float,              default=2e-6,   help='learning rate for v')
parser.add_argument('--A_lr', type=float,                       default=100 ,   help='learning rate for A')
# parser.add_argument('--learning_rate_min', type=float,          default=1e-8,   help='learning_rate_min')
# parser.add_argument('--decay', type=float,                      default=1e-3,   help='weight decay')
parser.add_argument('--beta1', type=float,                      default=0.9,    help='momentum')
parser.add_argument('--beta2', type=float,                      default=0.999,    help='momentum')
# parser.add_argument('--warm', type=float,                       default=10,    help='warmup step')
parser.add_argument('--num_step_lr', type=float,                default=10,    help='warmup step')
parser.add_argument('--decay_lr', type=float,                   default=1,    help='warmup step')
# parser.add_argument('--smoothing', type=float,                  default=0.1,    help='labelsmoothing')

parser.add_argument('--freeze', type=int,                       default=0,    help='whether freeze the pretrained encoder')

parser.add_argument('--traindata_loss_ratio', type=float,       default=0,    help='human translated data ratio')
parser.add_argument('--syndata_loss_ratio', type=float,         default=1,    help='augmented dataset ratio')

parser.add_argument('--valid_begin', type=int,                  default=1,      help='whether valid before train')
parser.add_argument('--train_A', type=int,                      default=1 ,     help='whether train A')
parser.add_argument('--attack', type=int,                       default=0 ,     help='whether att')

# parser.add_argument('--embedding_dim', type=int,                default=300 ,     help='whether train A')
parser.add_argument('--out_dim', type=int,                      default=2 ,     help='whether train A')
# parser.add_argument('--hidden_size', type=int,                  default=64 ,     help='whether train A')





args = parser.parse_args()#(args=['--batch_size', '8',  '--no_cuda'])#used in ipynb

args.test_num = args.test_num//args.batch_size * args.batch_size
args.train_w_num_points= args.train_w_num_points//args.w_bs * args.w_bs
args.train_A_num_points= args.train_A_num_points//args.A_bs * args.A_bs
args.unlabel_num_points= args.unlabel_num_points//args.syn_bs * args.syn_bs
args.rep_num = args.rep_num//args.batch_size * args.batch_size

args.test_num = args.train_w_num_points #TODO: test each epoch
args.rep_num = (args.train_w_num_points//4)//args.batch_size * args.batch_size#TODO: test each epoch

# %%
# https://wandb.ai/ check the running status online
import wandb
os.environ['WANDB_API_KEY'] = 'a166474b1b7ad33a0549adaaec19a2f6d3f91d87'
os.environ['WANDB_NAME'] = args.exp_name
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

wandb.init(project="Selftraining", config=args)


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


# Setting the seeds
np.random.seed(seed_)
torch.cuda.set_device(args.gpu)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True
torch.manual_seed(seed_)
cudnn.enabled = True
torch.cuda.manual_seed(seed_)


# %%

from datasets import load_dataset
l = ['dev','test','train','unlabeled']
dev = load_dataset('json', data_files='/tianyi-vol/yelp/dev_data.json', field='data')
test = load_dataset('json', data_files='/tianyi-vol/yelp/test_data.json', field='data')
train = load_dataset('json', data_files='/tianyi-vol/yelp/train_data.json', field='data')
unlabeled = load_dataset('json', data_files='/tianyi-vol/yelp/unlabeled_data.json', field='data')

# %%
print(train,dev,unlabeled,test)

# %%
from transformers import AutoTokenizer, AutoModelForMaskedLM
tokenizer = AutoTokenizer.from_pretrained("roberta-base")
modelname = args.model_name_teacher
pretrained = AutoModelForMaskedLM.from_pretrained(modelname)
pathname = modelname.replace('/', '')
logging.info(f'modelsize:{count_parameters_in_MB(pretrained)}MB')

if(exists(pathname+'.pt') == False):
    logging.info(f'saving to {pathname}')
    torch.save(pretrained, pathname+'.pt')

modelname = args.model_name_student
pretrained = AutoModelForMaskedLM.from_pretrained(modelname)
pathname = modelname.replace('/', '')
logging.info(f'modelsize:{count_parameters_in_MB(pretrained)}MB')
if(exists(pathname+'.pt') == False):
    logging.info(f'saving to {pathname}')
    torch.save(pretrained, pathname+'.pt')

modelname = args.model_name_de2en
pretrained = AutoModelForMaskedLM.from_pretrained(modelname)
pathname = modelname.replace('/', '')
logging.info(f'modelsize:{count_parameters_in_MB(pretrained)}MB')
if(exists(pathname+'.pt') == False):
    logging.info(f'saving to {pathname}')
    torch.save(pretrained, pathname+'.pt')


# %%

train =train["train"].shuffle(seed=seed_).select(range(args.train_w_num_points+args.train_A_num_points)) # A and W)
valid = dev["train"].shuffle(seed=seed_)#.select(range( r(args.valid_num_points, args.batch_size))) # dev
unlabeled = unlabeled["train"].shuffle(seed=seed_).select(range( args.unlabel_num_points) )# dev
test = test["train"].shuffle(seed=seed_)#.select(range( r(args.valid_num_points, args.batch_size))) # dev # test

logging.info("train len: %d", len(train))

train_w_num_points_len = args.train_w_num_points



train_v_synthetic_num_points_len = args.unlabel_num_points
train_A_num_points_len =  args.train_A_num_points

logging.info("train_w_num_points_len and train_v_num_points_len: %d", train_w_num_points_len)
logging.info("train_v_synthetic_num_points_len: %d",
             train_v_synthetic_num_points_len)
# logging.info("train_v_num_points_len: %d", train_v_num_points_len)
logging.info("train_A_num_points_len: %d", train_A_num_points_len)

attn_idx_list = torch.arange(train_w_num_points_len).cuda()
logging.info("valid len: %d", len(valid))
logging.info("test len: %d", len(test))
# logging.info(test[2])


# %%

# Create the DataLoader for our training set.
train_w_data = get_data_idx(train[:train_w_num_points_len], tokenizer,train_w_num_points_len)
train_A_data = get_data(train[train_w_num_points_len:], tokenizer)
train_syn_data = get_syn_data(unlabeled, tokenizer)

# indices = list(range(len(train)-train_w_num_points_len))

train_w_dataloader = DataLoader(train_w_data, sampler=SequentialSampler(train_w_data),
                              batch_size=args.w_bs, pin_memory=args.num_workers > 0, num_workers=args.num_workers)
logging.info(f'train w data size:{get_dataloader_size(train_w_dataloader)}')


train_syn_dataloader = DataLoader(train_syn_data, sampler=RandomSampler(train_syn_data),
                              batch_size=args.syn_bs, pin_memory=args.num_workers > 0, num_workers=args.num_workers)
logging.info(f'train syn data size:{get_dataloader_size(train_syn_dataloader)}')


train_A_dataloader = DataLoader(train_A_data,  sampler=RandomSampler(train_A_data),
                              batch_size=args.A_bs, pin_memory=args.num_workers > 0, num_workers=args.num_workers)
logging.info(f'train A data size:{get_dataloader_size(train_A_dataloader)}')



# Create the DataLoader for our training set.
valid_data = get_data(valid, tokenizer)
valid_dataloader = DataLoader(valid_data, sampler=SequentialSampler(valid_data),
                              batch_size=args.batch_size, pin_memory=args.num_workers > 0, num_workers=args.num_workers)
logging.info(f'validation data size:{get_dataloader_size(valid_dataloader)}')


# Create the DataLoader for our training set.
test_data = get_data(test, tokenizer)
test_dataloader = DataLoader(test_data, sampler=SequentialSampler(test_data),
                             batch_size=args.batch_size, pin_memory=args.num_workers > 0, num_workers=args.num_workers)  # , sampler=RandomSampler(test_data)
logging.info(f'test data size:{get_dataloader_size(test_dataloader)}')


# %%

A = attention_params(tokenizer, args, train_w_num_points_len)  # half of train regarded as u
A = A.cuda()

# TODO: model loaded from saved model
model_w = Model(tokenizer, args, 'teacher')
model_w = model_w.cuda()
w_optimizer = torch.optim.AdamW(model_w.parameters(
),  lr=args.w_lr,  betas=(args.beta1, args.beta2), eps=1e-8,weight_decay=1e-4)
# w_optimizer = Adafactor(model_w.parameters(), lr = args.w_lr ,scale_parameter=False, relative_step=False , warmup_init=False,clip_threshold=1,beta1=0,eps=( 1e-30,0.001))

scheduler_w = get_linear_schedule_with_warmup(w_optimizer, num_warmup_steps=args.epochs, num_training_steps=len(train_w_dataloader) * args.epochs)
# scheduler_w  = Scheduler(w_optimizer,dim_embed=512, warmup_steps=args.warm, initlr = args.w_lr)


model_v = Model(tokenizer, args, 'student')
model_v = model_v.cuda()
v_optimizer = torch.optim.AdamW(model_v.parameters(
),  lr=args.v_lr,  betas=(args.beta1, args.beta2), eps=1e-8,weight_decay=1e-4)
# v_optimizer =Adafactor(model_v.parameters(), lr = args.v_lr ,scale_parameter=False, relative_step=False , warmup_init=False,clip_threshold=1,beta1=0,eps=( 1e-30,0.001))

scheduler_v = get_linear_schedule_with_warmup(v_optimizer, num_warmup_steps=args.epochs, num_training_steps=len(train_w_dataloader) * args.epochs)
#  scheduler_v = StepLR(
    # v_optimizer, step_size=args.num_step_lr, gamma=args.decay_lr)
# scheduler_v  = Scheduler(v_optimizer,dim_embed=512, warmup_steps=args.warm, initlr = args.v_lr)


architect = Architect(model_w, model_v,  A, args)
architect.scheduler_A = get_linear_schedule_with_warmup(architect.optimizer_A, num_warmup_steps=args.epochs, num_training_steps=len(train_w_dataloader) * args.epochs)

# %%
@torch.no_grad()
def my_test(_dataloader,model,epoch):
    # logging.info(f"GPU mem before test:{getGPUMem(device)}%")
    acc = 0
    counter = 0
    model.eval()
    objs_top1 = AvgrageMeter()
    objs_top5 = AvgrageMeter()
    
    for step, batch in enumerate(_dataloader):
        test_dataloaderx = Variable(batch[0], requires_grad=False).to(device, non_blocking=False)
        test_dataloaderx_attn = Variable(batch[1], requires_grad=False).to(device, non_blocking=False)
        test_dataloadery = Variable(batch[2], requires_grad=False).to(device, non_blocking=False)
        logits,ls = my_loss(test_dataloaderx,test_dataloaderx_attn,test_dataloadery,model)
        n = test_dataloaderx.shape[0]
        acc+= ls.item()
        counter+= 1
        prec1, prec5 = accuracy(logits, test_dataloadery, topk=(1, 1))
                
        objs_top1.update(prec1.item(), n)
        
        objs_top5.update(prec5.item(), n)
    acc = objs_top1.avg
    logging.info('%s test loss : %f',model.name,acc/(counter))
    logging.info('%s top1 : %f',model.name,objs_top1.avg)
    objs_top1.reset()
    logging.info('%s top5 : %f',model.name,objs_top5.avg)
    objs_top5.reset()
    logging.info('%s test loss : %f',model.name,acc/(counter))
    wandb.log({'test_loss'+model.name: acc/counter})
    model.train()
    return acc

        

# %%
real_label = torch.zeros(train_w_num_points_len,device='cuda',dtype=torch.long)
def my_train(epoch, wdataloader,syndataloader,Adataloader, validdataloader, w_model, v_model, architect, A, w_optimizer, v_optimizer,  scheduler_w, scheduler_v, tot_iter, past_v_accu):
    objs_w = AvgrageMeter()
    objs_v_syn = AvgrageMeter()
    objs_v_train = AvgrageMeter()
    objs_v_star_val = AvgrageMeter()
    objs_v_val = AvgrageMeter()
    objs_w_top1 = AvgrageMeter()
    objs_w_top5 = AvgrageMeter()
    objs_v_top1 = AvgrageMeter()
    objs_v_top5 = AvgrageMeter()
    objs_weight = AvgrageMeter_tensor()
    improvementacc = 0
    w_trainloss_acc = 0
    # now  train_x is [num of batch, datasize], so its seperate batch for the code below
    wsize = args.w_bs
    synsize = args.syn_bs
    vsize = -1
    Asize = args.A_bs
    loader_len = len(wdataloader)
    w_model.train()
    v_model.train()

    for step, w_batch in enumerate(wdataloader):
        scheduler_w.step()
        scheduler_v.step()
        architect.scheduler_A.step()


        input_w = Variable(w_batch[0], requires_grad=False).to(
            device, non_blocking=False)
        input_w_attn = Variable(w_batch[1], requires_grad=False).to(
            device, non_blocking=False)
        output_w = Variable(w_batch[2], requires_grad=False).to(
            device, non_blocking=False)
        attn_idx = Variable(w_batch[3], requires_grad=False).to(
            device, non_blocking=False)
        real = Variable(w_batch[4], requires_grad=False).to(
            device, non_blocking=False)
        
        real_label[attn_idx] = real

        syn_batch = next(iter(syndataloader))
        input_syn = Variable(syn_batch[0], requires_grad=False).to(
            device, non_blocking=False)
        input_syn_attn = Variable(syn_batch[1], requires_grad=False).to(
            device, non_blocking=False)



        A_batch = next(iter(Adataloader))
        input_A_v = Variable(A_batch[0], requires_grad=False).to(
            device, non_blocking=False)
        input_A_v_attn = Variable(A_batch[1], requires_grad=False).to(
            device, non_blocking=False)
        output_A_v = Variable(A_batch[2], requires_grad=False).to(
            device, non_blocking=False)



        tot_iter[0] += input_w.shape[0]
        
        
        if(True):  # let v train on syn data and w data
            input_v = input_w
            input_v_attn = input_w_attn
            output_v = output_w
            vsize = wsize

        # input_w[:8,1:]= input_w[:8,1:] + torch.randint(0, 10, (input_w.shape[1]-1,),device='cuda')# noise input# bert would not learn from influent sentences

        v_star_val_loss=0
        if (args.train_A == 1 and epoch>=args.pre_epochs):
            epsilon_w = scheduler_w.get_lr()[0]
            epsilon_v  = scheduler_v.get_lr()[0]
            v_star_val_loss = architect.step(input_w,  output_w, input_w_attn, w_optimizer,
                                             input_v, input_v_attn, output_v, input_syn, input_syn_attn,
                                             input_A_v, input_A_v_attn, output_A_v, attn_idx,v_optimizer,
                                             epsilon_w, epsilon_v, args.grad_clip)
            objs_v_star_val.update(v_star_val_loss, Asize)

        with torch.no_grad():    
            objs_weight.update(A(input_w, input_w_attn, attn_idx).data)

        w_optimizer.zero_grad()
        logits, loss_w = CTG_loss(input_w, input_w_attn, output_w,
                                  A,attn_idx, w_model)
        w_trainloss_acc += loss_w.item()
        loss_w.backward()
        objs_w.update(loss_w.item(), wsize)
        w_optimizer.step()
        torch.nn.utils.clip_grad_norm(w_model.parameters(), args.grad_clip)
        prec1, prec5 = accuracy(logits, output_w, topk=(1, 1))
        objs_w_top1.update(prec1.item(), wsize)
        objs_w_top5.update(prec5.item(), wsize)

        if(epoch >= args.pre_epochs):
            v_optimizer.zero_grad()
            loss_aug = calc_loss_aug(
                input_syn, input_syn_attn, w_model, v_model)
            logits, loss = my_loss2(input_v, input_v_attn, output_v,
                                    v_model)
            v_loss = (args.traindata_loss_ratio*loss +
                      loss_aug*args.syndata_loss_ratio)
            v_loss.backward()
            objs_v_syn.update(loss_aug.item(), synsize)
            objs_v_train.update(loss.item(), vsize)
            v_optimizer.step()

            torch.nn.utils.clip_grad_norm(v_model.parameters(), args.grad_clip)
            prec1, prec5 = accuracy(logits, output_v, topk=(1, 1))
            objs_v_top1.update(prec1.item(), vsize)
            objs_v_top5.update(prec5.item(), vsize)


        with torch.no_grad():
            _,new_v_loss = my_loss2(
            input_A_v, input_A_v_attn,  output_A_v,model_v)
            improvementacc+=v_star_val_loss-new_v_loss.item()
            objs_v_val.update(new_v_loss.item(), Asize)


        progress = 100*(step)/(loader_len-1)

        
        if(tot_iter[0] % args.rep_num == 0 and tot_iter[0] != 0):
            logging.info('\n')
            logging.info(f"{progress:5.3}%:||W_train_loss:{objs_w.avg:^.7f}|V_train_syn_loss:{objs_v_syn.avg:^.7f}|V_train_loss:{objs_v_train.avg:^.7f}|V_val_loss:{objs_v_val.avg:^.7f}|V_star_val_loss:{objs_v_star_val.avg:^.7f}|improvement:{objs_v_star_val.avg-objs_v_val.avg:^.7f}|w_top1:{objs_w_top1.avg:^.7f}|w_top5:{objs_w_top5.avg:^.7f}|v_top1:{objs_v_top1.avg:^.7f}|v_top5:{objs_v_top5.avg:^.7f}|")
            temp = objs_weight.avg
            logging.info(f"avg weight:{temp}")
            logging.info(f"current alpha:{A.alpha[attn_idx].data}")
            logging.info(f"current weight:{A(input_w, input_w_attn, attn_idx)}")
            logging.info(f'noise:{torch.mean(temp[5:8]) if args.attack else None} mean:{torch.mean(temp)} max: {torch.max(temp)} min: {torch.min(temp)}')
            wandb.log({'W_train_loss': objs_w.avg})
            wandb.log({'V_train_syn_loss': objs_v_syn.avg})
            wandb.log({'V_train_loss': objs_v_train.avg})
            wandb.log({'V_star_val_loss': objs_v_star_val.avg})
            wandb.log({'V_val_loss': objs_v_star_val.avg})
            wandb.log({'W_accuracy': objs_w_top1.avg})
            wandb.log({'v_accuracy': objs_v_top1.avg})
            objs_v_syn.reset()
            objs_v_train.reset()
            objs_weight.reset()
            objs_w.reset()
            objs_v_star_val.reset()
            objs_v_val.reset()
            objs_w_top1.reset()
            objs_w_top5.reset()

        if(tot_iter[0] % args.test_num == 0 and tot_iter[0] != 0):
            w_accu = my_test(validdataloader, model_w, epoch)
            v_accu = my_test(validdataloader, model_v, epoch)
            wandb.log({'W_test_accuracy': w_accu})
            wandb.log({'v_test_accuracy':v_accu})
            torch.save(A, './model/'+'A.pt')
            if(v_accu>past_v_accu):
                past_v_accu = v_accu
                logging.info('find a better model')
                torch.save(model_w, './model/'+'model_w.pt')  # +now+
                torch.save(real_label, './model/'+'real_label.pt')
                torch.save(model_v, './model/'+'model_v.pt')
                torch.save(model_w.state_dict(), os.path.join(
                    wandb.run.dir, "model_w.pt"))
                torch.save(model_v.state_dict(), os.path.join(
                    wandb.run.dir, "model_v.pt"))
                torch.save(A.state_dict(), os.path.join(wandb.run.dir, "A.pt"))
                wandb.save("./files/*.pt", base_path="./files", policy="live")
            
            logging.info(f'current best accuracy:{past_v_accu}')
    logging.info(f'improvment:{improvementacc}')
    return w_trainloss_acc,past_v_accu


# %%
# if(args.valid_begin == 1):
#     my_test(valid_dataloader, model_w, -1)  # before train
#     my_test(valid_dataloader, model_v, -1)

tot_iter = [0]
v_accu = 0
for epoch in range(args.epochs):
    lr_w = scheduler_w.get_lr()[0]
    lr_v = scheduler_v.get_lr()[0]
    lr_A = architect.scheduler_A.get_lr()[0]

    logging.info(
        f"\n\n  ----------------epoch:{epoch},\t\tlr_w:{lr_w},\t\tlr_v:{lr_v},\t\tlr_A:{lr_A}----------------")

    w_train_loss,v_accu = my_train(epoch, train_w_dataloader,train_syn_dataloader,train_A_dataloader, valid_dataloader, model_w,
                            model_v,  architect, A, w_optimizer, v_optimizer, scheduler_w, scheduler_v, tot_iter,v_accu)

    # scheduler_w.step()
    # scheduler_v.step()
    # architect.scheduler_A.step()



w_accu = my_test(test_dataloader, torch.load('./model/'+'model_w.pt'), -2)
v_accu = my_test(test_dataloader, torch.load('./model/'+'model_v.pt'), -2)
logging.info(f'best w on test:{w_accu} accuracy; best v on test:{v_accu} accuracy')


# %%
real_label

# %%

w_accu = my_test(test_dataloader, torch.load('./model/'+'model_w.pt'), -2)
v_accu = my_test(test_dataloader, torch.load('./model/'+'model_v.pt'), -2)

logging.info(f'best w on test:{w_accu} accuracy; best v on test:{v_accu} accuracy')

# %%



