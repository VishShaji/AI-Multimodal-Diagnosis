import torch
import torch.nn as nn
from transformers import ViTModel
from models.base_model import BaseModel

class ViTModel(BaseModel):
    def __init__(self, pretrained_model, num_classes):
        super().__init__()
        #Load pretrained model
        self.vit = ViTModel.from_pretrained(pretrained_model)
        #Classifier layer to Transfor output to number of classes
        self.classifier = nn.Linear(self.vit.config.hidden_size, num_classes)
        self.freeze_vit()

    #Forward Pass
    def forward(self, x):
        #Forward propagation thorugh VIT
        outputs = self.vit(pixel_values=x)
        #Extracting CLS token
        logits = self.classifier(outputs.last_hidden_state[:, 0, :])
        return logits

    def predict(self, x):
        #Stopping Gradient Calculation and gets predicted output
        with torch.no_grad():
            outputs = self.forward(x)
            _, predicted = torch.max(outputs, 1)
        return predicted
    
    def freeze_vit(self):
        for param in self.vit.parameters():
            param.requires_grad = False

    def unfreeze_vit(self):
        for param in self.vit.parameters():
            param.requires_grad = True

    def get_fine_tuning_parameters(self):
        return [
            {'params': self.vit.parameters(), 'lr': 1e-5},
            {'params': self.classifier.parameters(), 'lr': 1e-3}
        ]