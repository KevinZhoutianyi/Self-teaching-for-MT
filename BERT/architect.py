import os
import gc
from losses import *
import random
import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from MT_hyperparams import *
from torch.optim.lr_scheduler import LambdaLR
import math

from torch.optim.lr_scheduler import StepLR



def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


seed_torch(seed_)


def _concat(xs):
    return torch.cat([x.view(-1) for x in xs])


class Architect(object):

    def __init__(self, w_model, v_model, A, args):


        self.beta1 = args.beta1
        self.beta2 = args.beta2

        self.w_model = w_model

        self.v_model = v_model
 
        self.args = args
        self.A = A
        self.param = list(filter(lambda x: x.requires_grad, self.A.parameters()))
        
        self.optimizer_A = torch.optim.SparseAdam(self.param,  lr=args.A_lr,betas=(0.5, 0.99), eps=1e-8)

        
        self.scheduler_A  =   StepLR(self.optimizer_A, step_size=args.num_step_lr, gamma=args.decay_lr)

    #########################################################################################
    # Computation of G' model named as unrolled model

    def _compute_unrolled_w_model(self, input, target, input_attn,  attn_idx,  eta_w, w_optimizer):
        # BART loss
        _,loss = CTG_loss(input, input_attn, target, 
                           self.A, attn_idx,self.w_model)
        # Unrolled model
#https://github.com/pytorch/pytorch/blob/master/torch/optim/adam.py,https://pytorch.org/docs/stable/generated/torch.optim.Adam.html

        theta = _concat(self.w_model.parameters()).data
        if(len(w_optimizer.state)==0):
            unrolled_w_model = self._construct_w_model_from_theta(theta)
            return unrolled_w_model
        
        step = w_optimizer.state[w_optimizer.param_groups[0]["params"][-1]]["step"]+1
        bias_correction1 = 1 - self.beta1 ** step
        bias_correction2 = 1 - self.beta2 ** step
        g = _concat(torch.autograd.grad(loss, self.w_model.parameters(), retain_graph=True))
        
        m = _concat(w_optimizer.state[v]['exp_avg']
                            for v in self.w_model.parameters()).mul(self.beta1) + (g).mul(1-self.beta1)
        v = _concat(w_optimizer.state[v]['exp_avg_sq']
                            for v in self.w_model.parameters()).mul(self.beta2) + (g*g.conj()).mul(1-self.beta2)
        denom = (v.sqrt() / math.sqrt(bias_correction2)).add(1e-08)
        step_size = eta_w / bias_correction1
        unrolled_w_model = self._construct_w_model_from_theta(
            theta.sub(step_size, torch.div(m,denom))
        )
        return unrolled_w_model

    # reshape the w model parameters
    def _construct_w_model_from_theta(self, theta):

        model_dict = self.w_model.state_dict()

        # create the new bart model
        w_model_new = self.w_model.new()

        # encoder update
        params, offset = {}, 0
        for k, v in self.w_model.named_parameters():
            v_length = np.prod(v.size())
            params[k] = theta[offset: offset+v_length].view(v.size())
            offset += v_length
        assert offset == len(theta)
        model_dict.update(params)
        w_model_new.load_state_dict(model_dict)

        return w_model_new

    def _compute_unrolled_v_model(self, input_v, input_v_attn, output_v, input_syn, input_syn_attn,  unrolled_w_model,  eta_v, v_optimizer):

        # DS loss on augmented dataset
        loss_aug = calc_loss_aug(
            input_syn, input_syn_attn, unrolled_w_model, self.v_model)
        _,loss = my_loss2(input_v, input_v_attn, output_v,
                         self.v_model)
        v_loss = (self.args.traindata_loss_ratio*loss+loss_aug *
                  self.args.syndata_loss_ratio)


        theta = _concat(self.v_model.parameters()).data
        if(len(v_optimizer.state)==0):
            unrolled_v_model = self._construct_v_model_from_theta(theta)
            return unrolled_v_model
        
        step =v_optimizer.state[v_optimizer.param_groups[0]["params"][-1]]["step"]+1
        bias_correction1 = 1 - self.beta1 ** step
        bias_correction2 = 1 - self.beta2 ** step
        g = _concat(torch.autograd.grad(v_loss, self.v_model.parameters(), retain_graph=True))
        
        m = _concat(v_optimizer.state[v]['exp_avg']
                            for v in self.v_model.parameters()).mul(self.beta1) + (g).mul(1-self.beta1)
        v = _concat(v_optimizer.state[v]['exp_avg_sq']
                            for v in self.v_model.parameters()).mul(self.beta2) + (g*g.conj()).mul(1-self.beta2)
        denom = (v.sqrt() / math.sqrt(bias_correction2)).add(1e-08)
        step_size = eta_v / bias_correction1
        unrolled_v_model = self._construct_v_model_from_theta(
            theta.sub(step_size, torch.div(m,denom))
        )
        
        return unrolled_v_model


    # reshape the T model parameters
    def _construct_v_model_from_theta(self, theta):

        model_dict = self.v_model.state_dict()

        # create the new bart model
        v_model_new = self.v_model.new()

        # encoder update
        params, offset = {}, 0
        for k, v in self.v_model.named_parameters():
            v_length = np.prod(v.size())
            params[k] = theta[offset: offset+v_length].view(v.size())
            offset += v_length

        # params['model.encoder.embed_tokens.weight'] = params['model.shared.weight']
        # params['embedding.embedding.weight'] = params['model.shared.weight']
        # params['model.decoder.embed_tokens.weight'] = params['model.shared.weight']
        assert offset == len(theta)
        model_dict.update(params)
        v_model_new.load_state_dict(model_dict)

        return v_model_new

    def step(self, input_w,  output_w, input_w_attn, w_optimizer,
                                             input_v, input_v_attn, output_v, input_syn, input_syn_attn,
                                             input_A_v, input_A_v_attn, output_A_v, attn_idx,v_optimizer,
                                             lr_w, lr_v):
             
        unrolled_w_model = self._compute_unrolled_w_model(
            input_w, output_w, input_w_attn, attn_idx,   lr_w, w_optimizer)
        unrolled_w_model.train() 
        # torch.nn.utils.clip_grad_norm(unrolled_w_model.parameters(), clip)

        unrolled_v_model = self._compute_unrolled_v_model(
            input_v, input_v_attn, output_v, input_syn, input_syn_attn, unrolled_w_model,  lr_v, v_optimizer)
        unrolled_v_model.train()
        # torch.nn.utils.clip_grad_norm(unrolled_v_model.parameters(), clip)
        _,unrolled_v_loss = my_loss2(
            input_A_v, input_A_v_attn,  output_A_v,unrolled_v_model)


        unrolled_v_loss.backward()
            
        vector_s_dash = [v.grad.data for v in unrolled_v_model.parameters()]

        implicit_grads_A = self._outer_A(vector_s_dash, input_w, output_w, input_w_attn,
                                         input_v, input_v_attn,  attn_idx,  unrolled_w_model, lr_w, lr_v)
        save(implicit_grads_A,'gradA')
        self.optimizer_A.zero_grad(set_to_none=True)
        for v, g in zip(self.param, implicit_grads_A):
            g = g.to_sparse()
            if v.grad is None:
                v.grad = Variable(g.data)#+v.data.to_sparse()*0.00001
            else:
                v.grad.data.copy_(g.data)#+v.data.to_sparse()*0.00001

        self.optimizer_A.step()

        del unrolled_w_model
        del unrolled_v_model
        gc.collect()
        return unrolled_v_loss.item()

    def _hessian_vector_product_A(self, vector, input, target, input_attn,  attn_idx,  r=1e-2):
        R = r / _concat(vector).norm()
        for p, v in zip(self.w_model.parameters(), vector):
        
            p.data = p.data.add(R, v)
        _,loss = CTG_loss(input, input_attn, target, 
                           self.A,attn_idx, self.w_model)

        # change to ctg dataset importance
        grads_p = torch.autograd.grad(loss, self.param)
        for p, v in zip(self.w_model.parameters(), vector):
            p.data = p.data.sub(2*R, v)
        _,loss = CTG_loss(input, input_attn, target,
                           self.A, attn_idx,self.w_model)

        # change to ctg dataset importance
        # change to .parameters()
        grads_n = torch.autograd.grad(loss, self.param)
        for p, v in zip(self.w_model.parameters(), vector):
            p.data = p.data.add(R, v)
        return [(x-y).div_(2*R) for x, y in zip(grads_p, grads_n)]

    ######################################################################
    # function for the product of hessians and the vector product wrt T and function for the product of
    # hessians and the vector product wrt G

    def _outer_A(self, vector_s_dash, w_input, w_target, w_input_attn, input_v, input_v_attn,  attn_idx,  unrolled_w_model, eta_w, eta_v, r=1e-2):
        # first finite difference method
        R1 = r / _concat(vector_s_dash).norm()
        for p, v in zip(self.v_model.parameters(), vector_s_dash):
            p.data = p.data.add(R1, v)

        loss_aug_p = calc_loss_aug(
            input_v, input_v_attn, unrolled_w_model, self.v_model)
        vector_dash = torch.autograd.grad(
            loss_aug_p, unrolled_w_model.parameters(), retain_graph=True)
        grad_part1 = self._hessian_vector_product_A(
            vector_dash, w_input, w_target, w_input_attn,attn_idx)

        # minus S
        for p, v in zip(self.v_model.parameters(), vector_s_dash):
            p.data = p.data.sub(2*R1, v)

        loss_aug_m = calc_loss_aug(
            input_v, input_v_attn, unrolled_w_model, self.v_model)

        vector_dash = torch.autograd.grad(
            loss_aug_m, unrolled_w_model.parameters(), retain_graph=True)

        grad_part2 = self._hessian_vector_product_A(
            vector_dash, w_input, w_target, w_input_attn,attn_idx)

        for p, v in zip(self.v_model.parameters(), vector_s_dash):
            p.data = p.data.add(R1, v)

        grad = [(x-y).div_((2*R1)/(eta_w*eta_v))
                for x, y in zip(grad_part1, grad_part2)]
        return grad

# # print("123")
