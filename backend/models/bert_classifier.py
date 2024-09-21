import torch
import torch.nn as nn
from transformers import BertModel
from models.base_model import BaseModel

class BERTClassifier(BaseModel):
    def __init__(self, pretrained_model, num_classes):
        super().__init__()
        self.bert = BertModel.from_pretrained(pretrained_model)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
        self.freeze_bert()

    #Forward Pass
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output

        #Applying Dropout and Classifier
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

    def predict(self, input_ids, attention_mask):
        #Stopping Gradient Calculation and gets predicted output
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask)
            _, predicted = torch.max(outputs, 1)
        return predicted
    
    def freeze_bert(self):
        for param in self.bert.parameters():
            param.requires_grad = False

    def unfreeze_bert(self):
        for param in self.bert.parameters():
            param.requires_grad = True

    def get_fine_tuning_parameters(self):
        return [
            {'params': self.bert.parameters(), 'lr': 1e-5},
            {'params': self.classifier.parameters(), 'lr': 1e-3}
        ]
