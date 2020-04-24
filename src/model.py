import torch
from torch import nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, filters, groups, keernel_size, padding):
        super(ResBlock, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(filters, filters, 1),
            nn.ReLU(),
            nn.GroupNorm(groups, filters),
            nn.Conv2d(filters, filters, keernel_size, padding=padding),
            nn.ReLU(),
            nn.GroupNorm(groups, filters),
            nn.Dropout(0.5),
            nn.Conv2d(filters, filters, 1),
            nn.ReLU(),
            nn.GroupNorm(groups, filters),
        )

    def forward(self, x):
        x_ = x
        x = self.layers(x)
        x = x + x_
        return x

class Model(nn.Module):
    def __init__(self, in_filters, image_size, filters, keernel_size, padding, num_res_blocks, num_classes):
        super(Model, self).__init__()
        h, w = image_size
        self.pre_layer = nn.Sequential(
            nn.Conv2d(in_filters, filters, keernel_size, padding=padding),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(filters, filters, keernel_size, padding=padding),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        layers = []
        for i in range(num_res_blocks):
            layers += [ResBlock(filters, 8, keernel_size, padding)]
        self.layers = nn.Sequential(*layers)
        self.classifier = nn.Linear(h * w * filters // 16, num_classes)

        self.init_weight()

    def init_weight(self):
        for n, p in self.named_parameters():
            if "weight" in n and  len(p.shape) > 2:
                torch.nn.init.kaiming_normal_(p)
            elif "bian" in n:
                torch.nn.init.zeros(p)

    def forward(self, x):
        b = x.shape[0]
        x = self.pre_layer(x)
        x = self.layers(x)
        x = x.reshape(b, -1)
        x = F.log_softmax(self.classifier(x), dim=-1)
        return x