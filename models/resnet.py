import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoImageProcessor, ResNetModel



class ResNet(nn.Module):

    def __init__(self):
        super(ResNet, self).__init__()
        self.processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
        self.model = ResNetModel.from_pretrained("microsoft/resnet-50")
        self.pooler = nn.AdaptiveMaxPool1d(1)
        self.head = nn.Linear(2048, 768)

    def forward(self, x, return_features=False):
        # sample first frame
        # x = x[0]
        # x = self.processor(x, return_tensors="pt")
        feat = self.model(x, output_hidden_states=True)
        pooled = self.pooler(feat.transpose(1, 2)).squeeze(2)
        return pooled, pooled
