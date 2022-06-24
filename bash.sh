# <!-- git clone https://github.com/KevinZhoutianyi/Self-teaching-for-machine-translation.git -->
cd ./Self-teaching-for-machine-translation
git pull
cd T5_newA

pip install torch==1.11.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
pip install --upgrade torchvision
pip install --upgrade torchtext



pip install transformers
pip install datasets
pip install sacrebleu==1.5.1
pip install torch_optimizer
pip install wandb
pip install fastt5    

