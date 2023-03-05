import torch.nn as nn
import torch
import torch.nn.functional as F
from ResNet import Residual1

class Encoder(nn.Module):
    def __init__(self,init_weights=False):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(3,64,kernel_size=7,padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            Residual1(64,1,init_weights=init_weights),
            Residual1(64,2,init_weights=init_weights),
            Residual1(128,2,init_weights=init_weights),
            Residual1(256,1,init_weights=init_weights),
        )
        if init_weights:
            self._initialize_weights()
    def forward(self,X):
        return self.layer(X)
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias,0)

class Classifier(nn.Module):
    def __init__(self,init_weights=False):
        super().__init__()
        self.layer = nn.Sequential(
            nn.AvgPool2d(kernel_size=4),
            nn.Flatten(),
            nn.Linear(256,256),
            nn.ReLU(inplace=True),
            nn.Linear(256,200)
        )
        if init_weights:
            self._initialize_weights()
    def forward(self,X):
        return self.layer(X)
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias,0)
            if isinstance(m,nn.Linear):
                nn.init.normal_(m.weight,0,0.01)
                nn.init.constant_(m.bias,0)

class Res(nn.Module):
    def __init__(self,inC=32,outC=32,init_weights=False):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(inC,outC,kernel_size=3,padding=1),
            nn.BatchNorm2d(outC),
            nn.ReLU(inplace=True)
        )
        if init_weights:
            self._initialize_weights()
    def forward(self,X):
        return F.relu(self.layer(X)+X)
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias,0)

class RevRes(nn.Module):
    def __init__(self,inC=32,outC=32,init_weights=False):
        super().__init__()
        self.layer = nn.Sequential(
            nn.ConvTranspose2d(inC,outC,kernel_size=4,padding=1,stride=2),
            nn.BatchNorm2d(outC),
            nn.ReLU(inplace=True)
        )
        if init_weights:
            self._initialize_weights()
    def forward(self,X):
        return self.layer(X)
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m,nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias,0)

class Decoder(nn.Module):
    def __init__(self,init_weights=False):
        super().__init__()
        self.layer = nn.Sequential(
            RevRes(256,256,init_weights=init_weights),
            # Res(256,256,init_weights=init_weights),
            RevRes(256,128,init_weights=init_weights),
            # Res(128,128,init_weights=init_weights),
            RevRes(128,64,init_weights=init_weights),
            # Res(64,64,init_weights=init_weights),
            RevRes(64,3,init_weights=init_weights),
            Res(3,3,init_weights=init_weights)
        )
        if init_weights:
            self._initialize_weights()
    def forward(self,X):
        return self.layer(X)
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m,nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias,0)

class AutoEncoder(nn.Module):
    def __init__(self,init_weights=False):
        super().__init__()
        self.encoder = Encoder(init_weights=init_weights)
        self.decoder = Decoder(init_weights=init_weights)
        self.classifier = Classifier(init_weights=init_weights)
        if init_weights:
            self._initialize_weights()
    def forward(self,X,mode='restru'):
        X = self.encoder(X)
        if mode == 'pred':
            return self.classifier(X)
        if mode == 'restru':
            return self.decoder(X)
        print('error!')
        return X
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias,0)
            if isinstance(m,nn.Linear):
                nn.init.normal_(m.weight,0,0.01)
                nn.init.constant_(m.bias,0)

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(device)
model = AutoEncoder(init_weights=True)
model = model.to(device)
X = torch.randn(64,3,64,64)
Y = model(X.to(device),mode='restru')
print(Y.shape)