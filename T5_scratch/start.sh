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
                --batch_size 256 --rep_num 100000 --test_num 1000000  --num_step_lr 1  --decay_lr 0.9 --num_workers 4\
                --train_w_num_points 64 --train_v_synthetic_num_points 128\
                --train_v_num_points 0 --train_A_num_points 64\
                --valid_begin 1 --train_A 1  --model_name_teacher google/t5-small-lm-adapt --model_name_student google/t5-small-lm-adapt\
                --w_lr 1e-3 --v_lr 1e-3 --A_lr 1e-3 --unrolled_w_lr 1e-3 --unrolled_v_lr 1e-3 \
                --exp_name BETA1.0

