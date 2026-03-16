
import torch
import torch.nn as nn
from torchvision import models

class DiseaseEnsemble(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.vgg = models.vgg16(pretrained=False)
        self.resnet = models.resnet50(pretrained=False)
        self.vgg.classifier[6] = nn.Linear(4096,128)
        self.resnet.fc = nn.Linear(2048,128)
        self.fc = nn.Linear(256,num_classes)
    def forward(self,x):
        v = self.vgg(x)
        r = self.resnet(x)
        x = torch.cat((v,r),dim=1)
        return self.fc(x)

class StageEnsemble(nn.Module):
    def __init__(self,num_classes=2):
        super().__init__()
        self.eff = models.efficientnet_b0(pretrained=False)
        self.mob = models.mobilenet_v3_small(pretrained=False)
        self.eff.classifier[1] = nn.Linear(1280,128)
        self.mob.classifier[3] = nn.Linear(1024,128)
        self.fc = nn.Linear(256,num_classes)
    def forward(self,x):
        e = self.eff(x)
        m = self.mob(x)
        x = torch.cat((e,m),dim=1)
        return self.fc(x)
