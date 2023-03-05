import torch.nn as nn
import torch
import torch.nn.functional as F

class MYBlock(nn.Module):
    def __init__(self,inC=3,outC=32,init_weights=False) :
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(inC,outC,kernel_size=3,padding=1,stride=1),
            nn.BatchNorm2d(outC),
            nn.ReLU()
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

class DenseNet(nn.Module):
    def __init__(self,init_weights=False):
        super().__init__()
        self.layer1 = MYBlock(init_weights=True)
        self.DownS = nn.MaxPool2d(kernel_size=2)
        self.layer2 = nn.Sequential(
            MYBlock(32,128,init_weights=True), # 输入和输出的通道数为参数
            MYBlock(128,128,init_weights=True),
            MYBlock(128,128,init_weights=True),
            MYBlock(128,128,init_weights=True)
        )
        self.layer3 = nn.Sequential(
            MYBlock(160,256,init_weights=True),
            MYBlock(256,256,init_weights=True),
            MYBlock(256,256,init_weights=True),
            MYBlock(256,256,init_weights=True)
        )
        self.layer4 = nn.Sequential(
            MYBlock(416,512,init_weights=True),
            MYBlock(512,512,init_weights=True),
            MYBlock(512,512,init_weights=True),
            MYBlock(512,512,init_weights=True)
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(928,200,kernel_size=1),
            nn.AvgPool2d(kernel_size=8)
        )
        if init_weights:
            self._initialize_weights()
    def forward(self,X):
        X = self.layer1(X)
        X1 = self.layer2(X)
        X = torch.cat((X,X1),1)
        X = self.DownS(X)
        X1 = self.layer3(X)
        X = torch.cat((X,X1),1)
        X = self.DownS(X)
        X1 = self.layer4(X)
        X = torch.cat((X,X1),1)
        X = self.DownS(X)
        X = self.layer5(X)
        X = X.squeeze(-1)
        X = X.squeeze(-1)
        return X
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias,0)

# if torch.cuda.is_available():
#     device = torch.device("cuda")
# else:
#     device = torch.device("cpu")
# print(device)
# model = DenseNet(init_weights=True)
# model = model.to(device)
# X = torch.randn(64,3,64,64)
# Y = model(X.to(device))
# print(Y.shape)