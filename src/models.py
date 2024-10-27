import torch
from torch import nn
from torchvision.models import VGG19_Weights
from torchvision import models

class NeuralStyleTransfer(nn.Module):
    def __init__(self):
        super().__init__()
        model = models.vgg19(weights=VGG19_Weights.DEFAULT)
        self.model = model.features
        self.freeze()

    def forward(self, x, layers):
        features = []
        for i, layer in self.model._modules.items():
            x = layer(x)
            if str(i) in layers:
                features.append(x)
        return features

    def freeze(self):
        for p in self.model.parameters():
            p.requires_grad = False