cd /tianyi-vol/Self-teaching-for-machine-translation/T5_scratch
pip install transformers
pip install datasets
pip install sacrebleu==1.5.1
pip install wandb
mkdir log
mkdir model
mkdir tensorboard
rm -f ./log/*.txt
rm -f ./tensorboard/*
python main.py --valid_num_points 2000 --train_num_points 508785 --valid_begin 1 --train_A 0

