import torch.nn as nn
import torch
import torch.nn.functional as F

class Residual1(nn.Module):
    def __init__(self,in_channels=64,rate=2,init_weights=False):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels,rate*in_channels,kernel_size=3,padding=1,stride=2),
            nn.BatchNorm2d(rate*in_channels,affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(rate*in_channels,rate*in_channels,kernel_size=3,padding=1,stride=1),
            nn.BatchNorm2d(rate*in_channels,affine=True)
        )
        self.layer1 = nn.Conv2d(in_channels,rate*in_channels,kernel_size=1,stride=2)
        if init_weights:
            self._initialize_weights()
    def forward(self,X):
        return F.relu(self.layer1(X)+self.layer(X))
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias,0)

class ResNet(nn.Module):
    def __init__(self,init_weights=False):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(3,64,kernel_size=7,padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            Residual1(64,1,init_weights=init_weights),
            Residual1(64,2,init_weights=init_weights),
            Residual1(128,2,init_weights=init_weights),
            Residual1(256,2,init_weights=init_weights),
            nn.AvgPool2d(kernel_size=4),
            nn.Flatten(),
            nn.Linear(512,200)
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

# if torch.cuda.is_available():
#     device = torch.device("cuda")
# else:
#     device = torch.device("cpu")
# print(device)
# model = ResNet(init_weights=True)
# model = model.to(device)
# X = torch.randn(64,3,56,56)
# Y = model(X.to(device))
# print(Y.shape)