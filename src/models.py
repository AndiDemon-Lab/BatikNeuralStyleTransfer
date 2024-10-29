import os
import torch
from torch import nn
from torchvision import models
from torchvision.models import (
    VGG19_Weights,
    Inception_V3_Weights,
    ResNet50_Weights,
    vgg19,
    inception_v3,
    resnet50,
)

class NeuralStyleTransfer(nn.Module):
    def __init__(self, model_name, pretrained_weights_path=None):
        super().__init__()
        self.model = self._create_model(model_name)
        self.freeze()
        
        # load pretrained weights if provided, else use default weights
        if pretrained_weights_path:
            self.load_weights(pretrained_weights_path)

    def _create_model(self, model_name):
        if model_name == "vgg19":
            model = vgg19(weights=VGG19_Weights.DEFAULT)
            return model.features
        elif model_name == "inception_v3":
            model = inception_v3(weights=Inception_V3_Weights.DEFAULT, aux_logits=True)
            # extract the layers we want from Inception v3
            return nn.Sequential(*list(model.children())[:-1])  # exclude the final classification layer
        elif model_name == "resnet50":
            model = resnet50(weights=ResNet50_Weights.DEFAULT)
            return nn.Sequential(*list(model.children())[:-2])  # use all layers except the final fully connected layers
        else:
            raise ValueError("Model name not recognized. Choose from 'vgg19', 'inception_v3', or 'resnet50'.")

    def forward(self, x, layers):
        features = []
        for i, layer in enumerate(self.model):
            x = layer(x)
            if str(i) in layers:
                features.append(x)
        return features

    def freeze(self):
        for p in self.model.parameters():
            p.requires_grad = False
            
    def load_weights(self, weights_path):
        """Load fine-tuned weights from the specified path."""
        if os.path.isfile(weights_path):
            state_dict = torch.load(weights_path, map_location='cpu')
            # Use strict=False to ignore any mismatched layers
            self.model.load_state_dict(state_dict, strict=False)
            print(f"Loaded weights from {weights_path}.")
        else:
            raise FileNotFoundError(f"No weight file found at {weights_path}.")
