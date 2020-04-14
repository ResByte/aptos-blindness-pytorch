# aptos-blindness-pytorch
Kaggle's Aptos Blindness Detection challenge with pytorch

## Requirements 
following needs to be installed befor running anything :
- torch
- torchvision 
- catalyst
- timm
- albumentations

## Run 

Modify config file in `config/config` to update parameters according to your needs

For training:
```shell
python train.py
```

Logs are stored under `./logs` folder with unique folder name. Tensorboard logs can be plotted as : 
```shell
tensorboard --logdir ${LOGDIR} --host 0.0.0.0
```

