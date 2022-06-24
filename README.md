
[basic idea](https://github.com/KevinZhoutianyi/Self-teaching-for-machine-translation/blob/main/ref/MT_self_teaching.pdf)

[math process](https://github.com/KevinZhoutianyi/Self-teaching-for-machine-translation/blob/main/ref/mt_math_withlambda.pdf)

## Code instruction:
W model -> teacher model

V model -> student model


# INSTRUCTION
- run bash.sh to install the requirements.
- T5_newA is the version which is currently working on
- run main.ipynb
- you need to change the wandb token in main or you can comment them out
- the best hyperparameters now are 
  >main.py --valid_num_points 3000 --train_num_points 1000000 --batch_size 270 --rep_num 50000 --test_num 1000000 --num_step_lr 1 --decay_lr 0.9 --num_workers 4 --train_w_num_points 64 --train_v_synthetic_num_points 128 --train_v_num_points 0 --train_A_num_points 78 --pre_epochs 0 --valid_begin 0 --train_A 1 --model_name_teacher google/t5-small-lm-adapt --model_name_student google/t5-small-lm-adapt --model_name_de2en Onlydrinkwater/t5-small-de-en-mt --w_lr 1e-3 --v_lr 1e-3 --A_lr 1e-4 --unrolled_w_lr 1e-3 --unrolled_v_lr 1e-3 --beta1 0.9 --beta2 0.98 --freeze 0 --exp_name BETA11,eval,noisedata,nofreeze,smallAlr
- here is the [log](https://wandb.ai/onlydrinkwater/Selftraining/runs/1pi3rin8/overview?workspace=user-onlydrinkwater)

