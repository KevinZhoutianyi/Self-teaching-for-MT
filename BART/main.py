# %%
import os
os.getcwd() 
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import warnings
warnings.filterwarnings("ignore")
from T5 import *
from datasets import load_dataset,load_metric
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
import transformers
import time
import argparse
from torch.utils.tensorboard import SummaryWriter
import string

# %%
parser = argparse.ArgumentParser("main")


parser.add_argument('--valid_num_points', type=int,             default = 1000, help='validation data number')
parser.add_argument('--train_num_points', type=int,             default = 4000, help='train data number')

parser.add_argument('--batch_size', type=int,                   default=16,     help='Batch size')
parser.add_argument('--train_w_num_points', type=int,           default=4,      help='train_w_num_points for each batch')
parser.add_argument('--train_w_synthetic_num_points', type=int, default=4,      help='train_w_synthetic_num_points for each batch')
parser.add_argument('--train_v_num_points', type=int,           default=4,      help='train_v_num_points for each batch')
parser.add_argument('--train_A_num_points', type=int,           default=4,      help='train_A_num_points decay for each batch')


parser.add_argument('--gpu', type=int,                          default=0,      help='gpu device id')
parser.add_argument('--epochs', type=int,                       default=50,     help='num of training epochs')
parser.add_argument('--pre_epochs', type=int,                   default=3,      help='train model W for x epoch first')
parser.add_argument('--grad_clip', type=float,                  default=5,      help='gradient clipping')

parser.add_argument('--w_lr', type=float,                       default=5e-6,   help='learning rate for w')
parser.add_argument('--v_lr', type=float,                       default=5e-6,   help='learning rate for v')
parser.add_argument('--A_lr', type=float,                       default=1e-4,   help='learning rate for A')
parser.add_argument('--learning_rate_min', type=float,          default=1e-5,   help='learning_rate_min')
parser.add_argument('--decay', type=float,                      default=1e-3,   help='weight decay')
parser.add_argument('--momentum', type=float,                   default=0.7,    help='momentum')


parser.add_argument('--traindata_loss_ratio', type=float,       default=0.5,    help='human translated data ratio')
parser.add_argument('--syndata_loss_ratio', type=float,         default=0.5,    help='augmented dataset ratio')

parser.add_argument('--valid_begin', type=int,                  default=1,      help='whether valid before train')
parser.add_argument('--train_A', type=int,                      default=0 ,     help='whether train A')


args = parser.parse_args()#(args=['--batch_size', '8',  '--no_cuda'])#used in ipynb

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
cudnn.benchmark = True
torch.manual_seed(seed_)
cudnn.enabled=True
torch.cuda.manual_seed(seed_)

# %%

pretrained  =  T5ForConditionalGeneration.from_pretrained("t5-small")
torch.save(pretrained,'T5BASE.pt')

# %%
# Load the tokenizer.
import random
tokenizer = T5Tokenizer.from_pretrained("t5-small")

criterion = torch.nn.CrossEntropyLoss( reduction='none')#,ignore_index = tokenizer.pad_token_id)#
# dataset = dataset.shuffle(seed=seed_)
train = dataset['train']['translation'][:args.train_num_points]
valid = dataset['validation']['translation'][:args.valid_num_points]
test = dataset['test']['translation']#[L_t+L_v:L_t+L_v+L_test]
def preprocess(dat):
    for t in dat:
        t['en'] = 'translate English to German: ' + t['en'] 
preprocess(train)
preprocess(valid)
preprocess(test)
num_batch = args.train_num_points//args.batch_size
train = train[:args.batch_size*num_batch]
logging.info("train len: %d",len(train))
train_w_num_points_len = num_batch * args.train_w_num_points
train_w_synthetic_num_points_len = num_batch * args.train_w_synthetic_num_points
train_v_num_points_len = num_batch * args.train_v_num_points
train_A_num_points_len = num_batch * args.train_A_num_points
logging.info("train_w_num_points_len: %d",train_w_num_points_len)
logging.info("train_w_synthetic_num_points_len: %d",train_w_synthetic_num_points_len)
logging.info("train_v_num_points_len: %d",train_v_num_points_len)
logging.info("train_A_num_points_len: %d",train_A_num_points_len)

attn_idx_list = torch.arange(train_w_num_points_len).cuda()
logging.info("valid len: %d",len(valid))
logging.info("test len: %d" ,len(test))
logging.info(train[2])

# %%
target_language  = 'de'
train_data = get_train_Dataset(train, tokenizer)# Create the DataLoader for our training set.
train_dataloader = DataLoader(train_data, sampler=SequentialSampler(train_data), 
                        batch_size=args.batch_size, pin_memory=True, num_workers=2)
valid_data = get_aux_dataset(valid, tokenizer)# Create the DataLoader for our training set.
valid_dataloader = DataLoader(valid_data, sampler=SequentialSampler(valid_data), 
                        batch_size=args.batch_size, pin_memory=True, num_workers=2)
test_data = get_aux_dataset(test, tokenizer)# Create the DataLoader for our training set.
test_dataloader = DataLoader(test_data, sampler=SequentialSampler(test_data),
                        batch_size=args.batch_size, pin_memory=True, num_workers=2)#, sampler=RandomSampler(test_data)

# %%

A = attention_params(train_w_num_points_len)#half of train regarded as u
A = A.cuda()

# TODO: model loaded from saved model
model_w = T5(criterion=criterion, tokenizer= tokenizer, name = 'model_w_in_main')
model_w = model_w.cuda()
w_optimizer = torch.optim.Adam(model_w.parameters(),args.w_lr)#,momentum=args.momentum,weight_decay=args.decay)
scheduler_w  = torch.optim.lr_scheduler.StepLR(w_optimizer,step_size=30, gamma=0.5)
# scheduler_w  = torch.optim.lr_scheduler.CosineAnnealingLR(w_optimizer, float(args.epochs), eta_min=args.learning_rate_min)



model_v = T5(criterion=criterion, tokenizer= tokenizer, name = 'model_v_in_main')
model_v = model_v.cuda()
v_optimizer = torch.optim.Adam(model_v.parameters(),args.v_lr)#,momentum=args.momentum,weight_decay=args.decay)
scheduler_v  = torch.optim.lr_scheduler.StepLR(v_optimizer,step_size=30, gamma=0.5)
# scheduler_v  = torch.optim.lr_scheduler.CosineAnnealingLR(v_optimizer, float(args.epochs), eta_min=args.learning_rate_min)



architect = Architect(model_w, model_v,  A, args)

# %%
# # x = ['im going to eat now ','it is my nameit is']
# # for index,i in enumerate(x) :
# #     x[index] = 'translate Enlgish to German:' + x[index]
# # y= tokenize(x, tokenizer, max_length = max_length)
# # input = y[0].cuda()
# # output  = model_v.generate(input,max_length=max_length)
# # tokenizer.batch_decode(output)


# metric_bleu =  load_metric('sacrebleu')
# predlist = ['Eine republikanische Strategie zur Bekämpfung der Wiederwahl Obamas','Die republikanischen Führer rechtfertigten ihre Politik mit der Notwendigkeit , Wahlbetrug zu bekämpfen .']
# targetlist = ['Eine republikanische Strategie um der Wiederwahl von Obama entgegenzutreten','Die Führungskräfte der Republikaner rechtfertigen ihre Politik mit der Notwendigkeit , den Wahlbetrug zu bekämpfen']
# predlist = ['Eine republikanische Strategie zur Bekämpfung der Wiederwahl Obamas', 'Die republikanischen Führer rechtfertigten ihre Politik mit der Notwendigkeit, Wahlbetrug zu bekämpfen.']
# targetlist =['Eine republikanische Strategie, um der Wiederwahl von Obama entgegenzutreten', 'Die Führungskräfte der Republikaner rechtfertigen ihre Politik mit der Notwendigkeit, den Wahlbetrug zu bekämpfen.']
# 
# predlist = [x.lower().translate( str.maketrans('', '', string.punctuation))  for x in predlist]
# targetlist = [[x.lower().translate( str.maketrans('', '', string.punctuation))] for x in targetlist]
# print(predlist)
# print(targetlist)
# metric_bleu.add_batch(predictions=predlist, references=targetlist)

# sacrebleu_score = metric_bleu.compute()
# print(sacrebleu_score)
# from nltk.translate import bleu
# from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
# def bleu(reference_captions, predicted_caption):
#     return 100 * sentence_bleu(reference_captions, predicted_caption,
#                                weights=(0.25, 0.25, 0.25,0.25), smoothing_function=SmoothingFunction().method1)
# x = bleu(targetlist[1],predlist[1])+bleu(targetlist[0],predlist[0])
# print(x/2)
                
                

# %%

from nltk.translate.bleu_score import sentence_bleu,corpus_bleu
def my_test(_dataloader,model,epoch):
    acc = 0
    counter = 0
    model.eval()
    metric_sacrebleu =  load_metric('sacrebleu')
    metric_bleu =  load_metric('bleu')

    for step, batch in enumerate(_dataloader):
        test_dataloaderx = Variable(batch[0], requires_grad=False).cuda()
        test_dataloaderx_attn = Variable(batch[1], requires_grad=False).cuda()
        test_dataloadery = Variable(batch[2], requires_grad=False).cuda()
        test_dataloadery_attn = Variable(batch[3], requires_grad=False).cuda()
        with torch.no_grad():
            ls = my_loss(test_dataloaderx,test_dataloaderx_attn,test_dataloadery,test_dataloadery_attn,model)
            acc+= ls
            counter+= 1
            pre = model.generate(test_dataloaderx)
            # print('pre',pre)
            try:
                x_decoded = tokenizer.batch_decode(test_dataloaderx,skip_special_tokens=True)
                pred_decoded = tokenizer.batch_decode(pre,skip_special_tokens=True)
                label_decoded =  tokenizer.batch_decode(test_dataloadery,skip_special_tokens=True)
                
                pred_str = [x.replace('.', '')  for x in pred_decoded]
                label_str = [[x.replace('.', '')] for x in label_decoded]
                pred_list = [x.replace('.', '').split()  for x in pred_decoded]
                label_list = [[x.replace('.', '').split()] for x in label_decoded]
                #pred_str = [x.translate( str.maketrans('', '', string.punctuation)) for x in pred_decoded] 
                # label_str = [[x.translate( str.maketrans('', '', string.punctuation))] for x in label_decoded]
                # pred_list = [x.translate( str.maketrans('', '', string.punctuation)).split()  for x in pred_decoded]#TODO:improve
                # label_list = [[x.translate( str.maketrans('', '', string.punctuation)).split()] for x in label_decoded]#TODO:improve
                if  step%100==0:
                    logging.info(f'x_decoded[:2]:{x_decoded[:2]}')
                    logging.info(f'pred_decoded[:2]:{pred_decoded[:2]}')
                    logging.info(f'label_decoded[:2]:{label_decoded[:2]}')
                metric_sacrebleu.add_batch(predictions=pred_str, references=label_str)
                metric_bleu.add_batch(predictions=pred_list, references=label_list)
                
               
            except Exception as ex:
                print(tokenizer.batch_decode(pre),[[x] for x in tokenizer.batch_decode(test_dataloadery)])
                raise Exception(ex)
        # logging.info(f"loss:{ls}")
    sacrebleu_score = metric_sacrebleu.compute()
    bleu_score = metric_bleu.compute()
    logging.info('%s sacreBLEU : %f',model.name,sacrebleu_score['score'])
    logging.info('%s BLEU : %f',model.name,bleu_score['bleu'])
    logging.info('%s test loss : %f',model.name,acc/(counter))
    writer.add_scalar(model.name+"/test_loss", acc/counter, global_step=epoch)
    writer.add_scalar(model.name+"/sacreBLEU",sacrebleu_score['score'], global_step=epoch)
    writer.add_scalar(model.name+"/BLEU",bleu_score['bleu'], global_step=epoch)
    model.train()
        

# %%
def my_train(epoch, _dataloader, w_model, v_model, architect, A, w_optimizer, v_optimizer, lr_w, lr_v, ):
    
    v_trainloss_acc = 0
    w_trainloss_acc = 0
    counter = 0
    wsize = args.train_w_num_points #now  train_x is [num of batch, datasize], so its seperate batch for the code below
    synsize = args.train_w_synthetic_num_points
    vsize = args.train_v_num_points 
    Asize = args.train_A_num_points 
    for step, batch in enumerate(_dataloader):
        counter+=1
        batch_loss_w, batch_loss_v = 0, 0
        
        train_x = Variable(batch[0], requires_grad=False).cuda()
        train_x_attn = Variable(batch[1], requires_grad=False).cuda()
        train_y = Variable(batch[2], requires_grad=False).cuda()
        train_y_attn = Variable(batch[3], requires_grad=False).cuda() 

        input_w = train_x[:wsize]
        
        input_w_attn = train_x_attn[:wsize]
        output_w = train_y[:wsize]
        output_w_attn = train_y_attn[:wsize]
        attn_idx = attn_idx_list[args.train_w_num_points*step:(args.train_w_num_points*step+args.train_w_num_points)]
           
        input_syn = train_x[wsize:wsize+synsize]
        input_syn_attn = train_x_attn[wsize:wsize+synsize]

        input_v = train_x[wsize+synsize:wsize+synsize+vsize]
        input_v_attn = train_x_attn[wsize+synsize:wsize+synsize+vsize]
        output_v = train_y[wsize+synsize:wsize+synsize+vsize]
        output_v_attn = train_y_attn[wsize+synsize:wsize+synsize+vsize]

        input_A_v      = train_x[wsize+synsize+vsize:wsize+synsize+vsize+Asize]
        input_A_v_attn = train_x_attn[wsize+synsize+vsize:wsize+synsize+vsize+Asize]
        output_A_v      = train_y[wsize+synsize+vsize:wsize+synsize+vsize+Asize]
        output_A_v_attn = train_y_attn[wsize+synsize+vsize:wsize+synsize+vsize+Asize]
       

        if (epoch <= args.epochs) and (args.train_A == 1) and epoch >= args.pre_epochs:
            architect.step(input_w,  output_w,input_w_attn, output_w_attn, w_optimizer, input_syn, input_syn_attn,input_A_v, input_A_v_attn, output_A_v, 
                output_A_v_attn, v_optimizer, attn_idx, lr_w, lr_v)
        
        if  epoch <= args.epochs and epoch <= args.epochs:
            
            w_optimizer.zero_grad()
            loss_w = CTG_loss(input_w, input_w_attn, output_w, output_w_attn, attn_idx, A, w_model)
            batch_loss_w += loss_w.item()
            w_trainloss_acc+=loss_w.item()
            loss_w.backward()
            # nn.utils.clip_grad_norm(w_model.parameters(), args.grad_clip)
            w_optimizer.step()
        # if epoch >= args.pre_epochs and epoch <= args.epochs:
        #     v_optimizer.zero_grad()
        #     loss_aug = calc_loss_aug(input_syn, input_syn_attn, w_model, v_model)#,input_v,input_v_attn,output_v,output_v_attn)
        #     loss = my_loss2(input_v,input_v_attn,output_v,output_v_attn,model_v)
            
        #     v_loss =  (args.syndata_loss_ratio*loss_aug+args.traindata_loss_ratio*loss)/num_batch
            
        #     batch_loss_v += v_loss.item()
        #     v_trainloss_acc+=v_loss.item()
        #     v_loss.backward()
        #     # nn.utils.clip_grad_norm(v_model.parameters(), args.grad_clip)
        #     v_optimizer.step()     
                
            
        if(step*args.batch_size%50==0):
            logging.info(f"{step*args.batch_size*100/(args.train_num_points)}%")
    logging.info(str(("Attention Weights A : ", A.alpha)))
    
    return w_trainloss_acc,v_trainloss_acc


# %%
if(args.valid_begin==1):
    my_test(valid_dataloader,model_w,-1) #before train
    my_test(valid_dataloader,model_v,-1)  
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

    
    my_test(valid_dataloader,model_w,epoch) 
    my_test(valid_dataloader,model_v,epoch)  

torch.save(model_v,'./model/'+now+'model_v.pt')
torch.save(model_v,'./model/'+now+'model_w.pt')
     
   
   
        
    



# %%


