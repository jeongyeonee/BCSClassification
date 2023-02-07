import torch
import torch.nn as nn
import timm
from efficientnet_pytorch import EfficientNet

class Classifier(nn.Module):
    def __init__(self, model_name, pretrained=True):
        super().__init__()
        
        if model_name == "b0":
            model_name = "tf_efficientnet_b0"
            
        elif model_name == "b1":
            model_name = "tf_efficientnet_b1"
        
        elif model_name == "b2":
            model_name = "tf_efficientnet_b2"
            
        elif model_name == "b3":
            model_name = "tf_efficientnet_b3" 
            
        self.model = timm.create_model(model_name, pretrained=pretrained, in_chans=3)
        self.n_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(self.n_features, 3)

    def forward(self, x):
        output = self.model(x)
        return output
