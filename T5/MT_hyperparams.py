import numpy as np
import torch

seed_ = 1

summary_length = 100

article_length = 500
valid_num_points = 200
train_num_points = 1000
test_num_points = 500


batch_size = 5
epochs = 500
ux_ratio = 0.3
learning_rate_min =0


w_lr = 0.001
v_lr = 0.0001
A_lr = 0.0005
begin_epoch = 0
momentum = 0.5
grad_clip =  5
decay = 1e-3