import os
import gc
from losses import *
import random
import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from MT_hyperparams import *

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

    def __init__(self, w_model, v_model, A):

        self.w_momentum = momentum
        self.w_decay = decay

        self.v_momentum =momentum
        self.v_decay = decay

        self.w_model = w_model

        self.v_model = v_model


        self.A = A

        # change to ctg dataset importance 
        # change to .parameters()

        self.optimizer_A = torch.optim.Adam(self.A.parameters(), 
          lr=A_lr, betas=(0.5, 0.999), weight_decay=decay)



    #########################################################################################
    # Computation of G' model named as unrolled model
    
    def _compute_unrolled_w_model(self, input, target, input_attn, target_attn, attn_idx, eta_w, w_optimizer):
        # BART loss
        loss = CTG_loss(input, input_attn, target, target_attn, attn_idx, self.A, self.w_model)
        # Unrolled model
        theta = _concat(self.w_model.parameters()).data
        logging.info(f"self.w_model.parameters(){self.w_model.parameters()}")
        logging.info(f"w_model{self.w_model}")
        dtheta = _concat(torch.autograd.grad(loss, self.w_model.parameters(), retain_graph = True )).data + self.w_decay*theta
        
        try:
            moment = _concat(w_optimizer.state[v]['momentum_buffer'] for v in self.w_model.parameters()).mul_(self.w_momentum)
        except:
            moment = torch.zeros_like(theta)
        # convert to the model
        unrolled_w_model = self._construct_w_model_from_theta(theta.sub(eta_w, moment+dtheta))
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

    # update the bart model with one step gradient update for unrolled model

    #########################################################################################
    
    #####################################################################################################
    # Computation of 'DS' model named as unrolled model
    
    def _compute_unrolled_v_model(self, input, input_attn,  unrolled_w_model,  eta_v, v_optimizer):

        # DS loss on augmented dataset
        loss = calc_loss_aug(input, input_attn, unrolled_w_model,self.v_model)

        # Unrolled model
        theta = _concat(self.v_model.parameters()).data
       
        dtheta = _concat(torch.autograd.grad(loss, self.v_model.parameters(), retain_graph = True )).data + self.v_decay*theta
        
        try:
            moment = _concat(v_optimizer.state[v]['momentum_buffer'] for v in self.v_model.parameters()).mul_(self.v_momentum)
        except:
            moment = torch.zeros_like(theta)
        # convert to the model
        unrolled_v_model = self._construct_v_model_from_theta(theta.sub(eta_v, dtheta+moment))

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

        assert offset == len(theta)
        model_dict.update(params)
        v_model_new.load_state_dict(model_dict)

        return v_model_new

    #########################################################################################

    # one step update for the importance parameter A
    def step(self, w_input, w_target, w_input_attn,  w_target_attn, w_optimizer,v_input, v_input_attn,valid_input_v, valid_input_v_attn, valid_out_v, 
                valid_out_v_attn, v_optimizer, attn_idx,  eta_w,eta_v):
        
        self.optimizer_A.zero_grad()
        # print("Self.A",self.A)
        # self.optimizer_B.zero_grad()
        # # print("architec shape:",w_input.shape, w_target.shape, w_input_attn.shape, w_target_attn.shape, attn_idx.shape)
        unrolled_w_model = self._compute_unrolled_w_model(w_input, w_target, w_input_attn, w_target_attn, attn_idx, eta_w, w_optimizer)

        unrolled_w_model.train()

        # unrolled_bartrt_model.bart_model.train()

        unrolled_v_model = self._compute_unrolled_v_model(v_input, v_input_attn,  unrolled_w_model,  eta_v, v_optimizer)

        unrolled_v_loss = unrolled_v_model( valid_input_v, valid_input_v_attn,  valid_out_v, valid_out_v_attn).loss

        unrolled_v_model.train()

        unrolled_v_loss.backward()

        vector_s_dash = [v.grad.data for v in unrolled_v_model.parameters()]


        implicit_grads_A = self._outer_A(vector_s_dash, w_input, w_target, w_input_attn,  w_target_attn, v_input, v_input_attn, attn_idx, unrolled_w_model, eta_w, eta_v) 


    

        # # change to ctg dataset importance
        # # change to .parameters()
        # print(self.A.parameters())
        # print(implicit_grads_A)
        for v, g in zip(self.A.parameters(), implicit_grads_A):
            #print(g.data)tensor([1.0071e+09, 1.0071e+09, 1.0071e+09,  ..., 1.0071e+09, 1.0071e+09,
            # print(v)
            if v.grad is None:
                v.grad = Variable(g.data)
            else:
                v.grad.data.copy_(g.data)

        self.optimizer_A.step()
        # TODO: A.alpha got same value for all training data
        del unrolled_w_model

        del unrolled_v_model

        gc.collect()

    ######################################################################

    def _hessian_vector_product_A(self, vector, input, target, input_attn, target_attn, attn_idx, r=1e-2):
        R = r / _concat(vector).norm()
        for p, v in zip(self.w_model.parameters(), vector):
            p.data.add_(R, v)
        loss = CTG_loss(input, input_attn, target, target_attn, attn_idx, self.A, self.w_model)

        # change to ctg dataset importance
        grads_p = torch.autograd.grad(loss, self.A.parameters())
        # print("grads_p",type(grads_p))
        # print("grads_p",(grads_p))
        # print("grads_p",torch.max(grads_p[0]))
        # print("grads_p",torch.min(grads_p[0]))
        for p, v in zip(self.w_model.parameters(), vector):
            p.data.sub_(2*R, v)
        loss = CTG_loss(input, input_attn, target, target_attn, attn_idx, self.A, self.w_model)

        # change to ctg dataset importance
        # change to .parameters()
        grads_n = torch.autograd.grad(loss, self.A.parameters())

        for p, v in zip(self.w_model.parameters(), vector):
            p.data.add_(R, v)

        return [(x-y).div_(2*R) for x, y in zip(grads_p, grads_n)]


    ######################################################################
    # function for the product of hessians and the vector product wrt T and function for the product of
    # hessians and the vector product wrt G
    def _outer_A(self, vector_s_dash, w_input, w_target, w_input_attn,  w_target_attn, input_v, input_v_attn, attn_idx, unrolled_w_model, eta_w, eta_v, r=1e-2):
        #first finite difference method
        R1 = r / _concat(vector_s_dash).norm()
        for p, v in zip(self.v_model.parameters(), vector_s_dash):
            
            p.data.add_(R1, v)
        unrolled_w_model.train()

        loss_aug_p = calc_loss_aug(input_v, input_v_attn, unrolled_w_model, self.v_model)


        vector_dash = torch.autograd.grad(loss_aug_p, unrolled_w_model.parameters(), retain_graph = True)

        grad_part1 = self._hessian_vector_product_A(vector_dash, w_input, w_target, w_input_attn, w_target_attn, attn_idx)

        # minus S
        for p, v in zip(self.v_model.parameters(), vector_s_dash):
            p.data.sub_(2*R1, v)

        loss_aug_m = calc_loss_aug(input_v, input_v_attn, unrolled_w_model, self.v_model)
        
        # T

        vector_dash = torch.autograd.grad(loss_aug_m, unrolled_w_model.parameters(), retain_graph = True)

        grad_part2 = self._hessian_vector_product_A(vector_dash, w_input, w_target, w_input_attn, w_target_attn , attn_idx)

        for p, v in zip(self.v_model.parameters(), vector_s_dash):
            p.data.add_(R1, v)

        grad = [(x-y).div_((2*R1)/(eta_w*eta_v)) for x, y in zip(grad_part1, grad_part2)]

        return grad

# # print("123")