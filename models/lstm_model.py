import torch
import torch.nn as nn
from transformers import AutoModel
from models.base_model import BaseModel

class LSTMModel(BaseModel):
    def __init__(self, pretrained_model, hidden_size, num_layers, output_size, dropout_rate):
        super().__init__()
        self.bert = AutoModel.from_pretrained(pretrained_model)
        self.lstm = nn.LSTM(
            self.bert.config.hidden_size, 
            hidden_size, 
            num_layers, 
            batch_first=True, 
            dropout=dropout_rate if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size, output_size)
        self.freeze_bert()

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        lstm_output, _ = self.lstm(bert_outputs.last_hidden_state)
        lstm_output = self.dropout(lstm_output[:, -1, :])
        return self.fc(lstm_output)

    def predict(self, x):
        with torch.no_grad():
            outputs = self.forward(x)
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
            {'params': self.lstm.parameters(), 'lr': 1e-4},
            {'params': self.fc.parameters(), 'lr': 1e-3}
        ]