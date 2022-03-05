cd /tianyi-vol/Self-teaching-for-machine-translation/T5
pip install transformers
pip install datasets
pip install sacrebleu==1.5.1
mkdir log
python main.py --valid_num_points 2000 --train_num_points 30000 --valid_begin 1

