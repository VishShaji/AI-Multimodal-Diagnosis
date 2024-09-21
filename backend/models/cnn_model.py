import torch
import torch.nn as nn
import torchvision.models as models
from models.base_model import BaseModel

class CheXNet(nn.Module):
    def __init__(self, num_classes):
        super(CheXNet, self).__init__()
        # Use DenseNet as base model
        self.base_model = models.densenet121(pretrained=True)  
        num_ftrs = self.base_model.classifier.in_features
        self.base_model.classifier = nn.Sequential(
            nn.Linear(num_ftrs, num_classes),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.base_model(x)

class CNNModel(BaseModel):
    def __init__(self, num_classes, weight_path):
        super().__init__()
        self.chexnet = CheXNet(num_classes)

        # Load pretrained weights
        self.chexnet.load_state_dict(torch.load(weight_path))
        self.freeze_base_model()

    def forward(self, x):
        return self.chexnet(x)

    def predict(self, x):
        with torch.no_grad():
            outputs = self.forward(x)
            predicted = (outputs > 0.5).float()
        return predicted
    
    def freeze_base_model(self):
        for param in self.chexnet.base_model.parameters():
            param.requires_grad = False

    def unfreeze_base_model(self):
        for param in self.chexnet.base_model.parameters():
            param.requires_grad = True

    def get_fine_tuning_parameters(self):
        return [
            {'params': self.chexnet.base_model.features.parameters(), 'lr': 1e-4},
            {'params': self.chexnet.base_model.classifier.parameters(), 'lr': 1e-3}
        ]
