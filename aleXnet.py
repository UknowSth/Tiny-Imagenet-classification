import torch.nn as nn
import torch

class Alexnet(nn.Module):
    def __init__(self,num_classes=200,init_weights=False):
        super(Alexnet,self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,48,kernel_size=11,stride=4,padding=2), # output [48,55,55]
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2),              # output [48,27,27]

            nn.Conv2d(48,128,kernel_size=5,padding=2),         # output [128,27,27]
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2),              # output [128,13,13]

            nn.Conv2d(128,192,kernel_size=3,padding=1),        # output [192,13,13]
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),

            nn.Conv2d(192,192,kernel_size=3,padding=1),        # output [192,13,13]
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),

            nn.Conv2d(192,128,kernel_size=3,padding=1),        # output [128,13,13]
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2),              # output [128,6,6]
            nn.Flatten(),
        )
        self.classfier = nn.Sequential(
            # nn.Dropout(p=0.5),
            nn.Linear(128*6*6,2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),

            # nn.Dropout(p=0.5),
            nn.Linear(2048,1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),

            nn.Linear(1024,num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self,X):
        X = self.features(X)
        X = self.classfier(X)
        return X
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m,(nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias,0)
            elif isinstance(m,nn.Linear):
                nn.init.normal_(m.weight,0,0.01)
                nn.init.constant_(m.bias,0)