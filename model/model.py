import torch 
import torch.nn as nn
import torchvision

class AptosModel(nn.Module):
    def __init__(self, 
                 arch:str='resnet50', 
                 z_dims:int=512, 
                 nb_classes:int=5, 
                 freeze:bool=True,
                 drop:float=0.5):
        super(AptosModel, self).__init__()
        self.freeze = freeze
        
        # feature extractor , all layers until Adaptive average pooling
        backbone = list(torchvision.models.__dict__[arch](pretrained=True).children())[:-1]
        in_features = torchvision.models.__dict__[arch](pretrained=False).fc.in_features
        self.features = nn.Sequential(*backbone)
        if self.freeze:
            self.features.eval()
        self.classifier = nn.Sequential(
            nn.Linear(in_features, z_dims),
            nn.Dropout(drop),
            nn.Linear(z_dims, nb_classes)
        )

    def forward(self, x):
        if self.freeze:
            with torch.no_grad():
                x = self.features(x)
        else:
            x = self.features(x)
        x = self.classifier(x.squeeze())
        return x