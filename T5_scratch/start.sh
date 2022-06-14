cd /tianyi-vol/Self-teaching-for-machine-translation/T5_scratch
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
python main.py --valid_num_points 3000 --train_num_points 3000000 \
                --batch_size 400 --rep_num 100000 --test_num 1000000  --num_step_lr 1 --num_workers 4\
                --train_w_num_points 400 --train_v_synthetic_num_points 0\
                --train_v_num_points 0 --train_A_num_points 0\
                --valid_begin 1 --train_A 1  --model_name_teacher google/t5-small-lm-adapt --model_name_student t5-small\
                --w_lr 1e-3 --v_lr 0 --A_lr 0 --unrolled_w_lr 0 --unrolled_v_lr 0 \
                --exp_name server,3kk,lm

