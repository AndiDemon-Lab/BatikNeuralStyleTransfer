import os
import torch
from torch import nn
from torchvision.models import (
    VGG19_Weights,
    Inception_V3_Weights,
    ResNet50_Weights,
    ResNet101_Weights,
    VGG16_Weights,
    vgg19,
    vgg16,
    inception_v3,
    resnet50,
    resnet101
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
            # Load VGG19 model with pretrained weights
            model = vgg19(weights=VGG19_Weights.IMAGENET1K_V1)
            return model.features

        elif model_name == "vgg16":
            # Load VGG16 model with pretrained weights
            model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
            return model.features

        elif model_name == "inception_v3":
            # Load Inception V3 model with pretrained weights
            model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, aux_logits=False)
            # Exclude final classification layer
            return nn.Sequential(*list(model.children())[:-1])

        elif model_name == "resnet50":
            # Load ResNet50 model with pretrained weights
            model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            # Exclude fully connected and pooling layers
            return nn.Sequential(*list(model.children())[:-2])

        elif model_name == "resnet101":
            # Load ResNet101 model with pretrained weights
            model = resnet101(weights=ResNet101_Weights.IMAGENET1K_V1)
            # Exclude fully connected and pooling layers
            return nn.Sequential(*list(model.children())[:-2])

        else:
            raise ValueError("Model name not recognized. Choose from 'vgg19', 'vgg16', 'inception_v3', 'resnet50', or 'resnet101'.")

    def forward(self, x, layers):
        features = []
        layers = set(map(int, layers))
        for i, (name, layer) in enumerate(self.model._modules.items()):
            x = layer(x)
            if i in layers:
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
