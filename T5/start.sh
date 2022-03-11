cd /tianyi-vol/Self-teaching-for-machine-translation/T5
pip install transformers
pip install datasets
pip install sacrebleu==1.5.1
mkdir log
mkdir model
mkdir tensorboard
python main.py --valid_num_points 200 --train_num_points 300 --valid_begin 1

