import torch.nn as nn
import torch
import torch.nn.functional as F

class Stem(nn.Module):
    def __init__(self,init_weights=False):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3,32,kernel_size=3,stride=2), # output [32,27,27]
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32,32,kernel_size=3,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32,64,kernel_size=3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.layer2z = nn.MaxPool2d(kernel_size=3,stride=2)
        self.layer2y =  nn.Sequential(
            nn.Conv2d(64,96,kernel_size=3,stride=2),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True)
        )
        self.layer3z = nn.Sequential(
            nn.Conv2d(160,64,kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,96,kernel_size=3,padding=1,stride=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True)
        )
        self.layer3y = nn.Sequential(
            nn.Conv2d(160,64,kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,64,kernel_size=(7,1),padding=(3,0)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,64,kernel_size=(1,7),padding=(0,3)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,96,kernel_size=3,padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True)
        )
        self.layer4z = nn.MaxPool2d(kernel_size=3,stride=2)
        self.layer4y = nn.Sequential(
            nn.Conv2d(192,192,kernel_size=3,stride=2),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True)
        )
        if init_weights:
            self._initialize_weights()
    def forward(self,X):
        X = self.layer1(X)
        X1 = self.layer2z(X)
        X2 = self.layer2y(X)
        X = torch.cat((X1,X2),1)
        X1 = self.layer3z(X)
        X2 = self.layer3y(X)
        X = torch.cat((X1,X2),1)
        X1 = self.layer4z(X)
        X2 = self.layer4y(X)
        return torch.cat((X1,X2),1)
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias,0)

class InceptionA(nn.Module):
    def __init__(self,init_weights=False,rate=1.0): # rate=1.0
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(384,32,kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32,48,kernel_size=3,padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.Conv2d(48,64,kernel_size=3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,384,kernel_size=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(384,32,kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32,32,kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(384,32,kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(448,384,kernel_size=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True)
        )
        self.rate = rate
        if init_weights:
            self._initialize_weights()
    def forward(self,X):
        X = F.relu(X)
        X1 = self.layer1(X)
        X2 = self.layer2(X)
        X3 = self.layer3(X)
        X4 = torch.cat((X1,X2,X3),1)
        X4 = self.layer4(X4)
        return F.relu(X + self.rate*X4) # self.rate*
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias,0)

class Reduction(nn.Module):
    def __init__(self,init_weights=False):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(384,256,kernel_size=3,stride=2,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,256,kernel_size=3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True), 
            nn.Conv2d(256,384,kernel_size=3,padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(384,384,kernel_size=3,stride=2,padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True)
        )
        self.layer3 = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        if init_weights:
            self._initialize_weights()
    def forward(self,X):
        X1 = self.layer1(X)
        X2 = self.layer2(X)
        X3 = self.layer3(X)
        return torch.cat((X1,X2,X3),1)
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias,0)

class InceptionB(nn.Module):
    def __init__(self,init_weights=False,rate=1.0):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1152,128,kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128,160,kernel_size=(1,7),padding=(0,3)),
            nn.BatchNorm2d(160),
            nn.ReLU(inplace=True),
            nn.Conv2d(160,192,kernel_size=(7,1),padding=(3,0)),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(1152,192,kernel_size=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(384,1152,kernel_size=1),
            nn.BatchNorm2d(1152),
            nn.ReLU(inplace=True)
        )
        self.rate = rate
        if init_weights:
            self._initialize_weights()
    def forward(self,X):
        X = F.relu(X)
        X1 = self.layer1(X)
        X2 = self.layer2(X)
        X3 = torch.cat((X1,X2),1)
        X3 = self.layer3(X3)
        return F.relu(X + self.rate*X3)
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias,0)

class Inception_ResNet(nn.Module):
    def __init__(self,init_weights=False):
        super().__init__()
        self.layer = nn.Sequential(
            Stem(init_weights=init_weights),
            # 目前暂时取n=1
            InceptionA(init_weights=init_weights,rate=0.1),
            Reduction(init_weights=init_weights),
            InceptionB(init_weights=init_weights,rate=0.1),
            InceptionB(init_weights=init_weights,rate=0.1),
            nn.AvgPool2d(kernel_size=3),
            nn.Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(1152,200)
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
                

class Inception_ResNet3(nn.Module):
    def __init__(self,init_weights=False):
        super().__init__()
        self.layer = nn.Sequential(
            Stem(init_weights=init_weights),
            InceptionA(init_weights=init_weights),
            Reduction(init_weights=init_weights),
            # InceptionB(init_weights=init_weights),
            InceptionB(init_weights=init_weights,rate=0.1),
            nn.AvgPool2d(kernel_size=3),
            nn.Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(1152,200)
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

class Inception_ResNet2(nn.Module): # 取n=2构造模型
    def __init__(self,init_weights=False):
        super().__init__()
        self.layer = nn.Sequential(
            Stem(init_weights=init_weights),
            InceptionA(init_weights=init_weights,rate=0.1),
            InceptionA(init_weights=init_weights,rate=0.1),
            Reduction(init_weights=init_weights),
            InceptionB(init_weights=init_weights,rate=0.1),
            InceptionB(init_weights=init_weights,rate=0.1),
            InceptionB(init_weights=init_weights,rate=0.1),
            InceptionB(init_weights=init_weights,rate=0.1),
            nn.AvgPool2d(kernel_size=3),
            nn.Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(1152,200)
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
# model = Inception_ResNet2(init_weights=True)
# model = model.to(device)
# X = torch.randn(64,3,56,56)
# Y = model(X.to(device))
# print(Y.shape)
        