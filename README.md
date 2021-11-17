# Fedrated Learning With Domain Generalization
Code to reproduce the experiments of **Federated Adversarial Domain Generalization**.
## How to use it
* Clone or download the repository
### Install the requirement
 >  pip install -r requirements.txt
### Download VLCS and PACS
* Download VLCS from https://github.com/jindongwang/transferlearning/blob/master/data/dataset.md, extract, move it to ./data/VLCS/
* Download PACS from https://drive.google.com/uc?id=0B6x7gtvErXgfbF9CSk53UkRxVzg, move it to ./data/PACS/
* Download pre-trained AlexNet from https://drive.google.com/file/d/1wUJTH1Joq2KAgrUDeKJghP1Wf7Q9w4z-/view?usp=sharing and move it to ./data/
### Running ours on VLCS
``` 
cd src
python train.py --lr0 0.001 --lr1 0.0007 --label_smoothing 0.2 --lr-threshold 0.0001 --factor 0.2 --epochs 10 --rp-size 1024 --patience 20 --ite-warmup 500 --global_epochs  20
```
### Running ours on PACS
``` 
cd pacs-ours
python train.py --lr0 0.01 --lr1 0.007 --label_smoothing 0.01 --lr-threshold 0.0001 --factor 0.2 --epochs 7 --i-epochs 3 --rp-size 1024 --patience 20 --ite-warmup 100 --global_epochs  30
```


