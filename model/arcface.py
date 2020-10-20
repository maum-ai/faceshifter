import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet101


class Linear(nn.Module):
    __constants__ = ['bias', 'in_features', 'out_features']

    def __init__(self, in_features, out_features, bias=False):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        return F.linear(input, F.normalize(self.weight, p=2, dim=1), self.bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class Arcface(nn.Module):
    def __init__(self, vector_size=256, num_classes=360576):
        super(Arcface, self).__init__()
        self.resnet = resnet101(pretrained=False, num_classes=vector_size)
        self.fc = Linear(256, num_classes, bias=False)

        self.m = 0.5
        self.s = 64.0
        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)

        self.mm = self.sin_m * self.m * self.s
        self.th = math.cos(math.pi - self.m) * self.s

    def forward(self, x, label):  # B, 3, H, W
        x = self.resnet(x)  # B, 256
        x = F.normalize(x, p=2, dim=1) * self.s
        cosine = self.fc(x).clamp(-self.s, self.s)  # B, 1

        sine = torch.sqrt((self.s * self.s - torch.pow(cosine, 2)).clamp(0, self.s))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine >= self.th, phi, cosine - self.mm)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        return output