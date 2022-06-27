cd /tianyi-vol/Self-teaching-for-machine-translation/T5_newA
ulimit -n 50000
pip install transformers
pip install datasets
pip install sacrebleu==1.5.1
pip install torch_optimizer
pip install wandb
pip install fastt5    
mkdir log
mkdir model
mkdir tensorboard
rm -f ./log/*.txt
rm -f ./tensorboard/*
python main.py --valid_num_points 3000 --train_num_points 1000000 --batch_size 270 --rep_num 50000 --test_num 1000000 \
--num_step_lr 1 --decay_lr 0.9 --num_workers 4\
 --train_w_num_points 64 --train_v_synthetic_num_points 128 --train_v_num_points 0 --train_A_num_points 78\
  --pre_epochs 0 --valid_begin 0 --train_A 1 --traindata_loss_ratio 0 --syndata_loss_ratio 1\
   --model_name_teacher google/t5-small-lm-adapt --model_name_student google/t5-small-lm-adapt --model_name_de2en Onlydrinkwater/t5-small-de-en-mt\
    --w_lr 1e-3 --v_lr 1e-3 --A_lr 0 --unrolled_w_lr 1e-3 --unrolled_v_lr 1e-3 --beta1 0.9 --beta2 0.98 --freeze 0\
     --exp_name BETA13,baseline,noA
