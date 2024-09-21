import torch
import torch.nn as nn
from ts2vec.ts2vec import TS2Vec
from models.base_model import BaseModel

class LSTMModel(BaseModel):
    def __init__(self, input_dims, hidden_size, output_size, dropout_rate):
        super().__init__()
        self.ts2vec = TS2Vec(input_dims=input_dims)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input_data):
        # Pass the input time-series data to TS2Vec model
        ts_embeddings = self.ts2vec(input_data)

        # Final layers
        ts_embeddings = self.dropout(ts_embeddings[:, -1, :])
        return self.fc(ts_embeddings)

    def predict(self, x):
        with torch.no_grad():
            outputs = self.forward(x)
            _, predicted = torch.max(outputs, 1)
        return predicted

    def get_fine_tuning_parameters(self):
        return [
            {'params': self.ts2vec.parameters(), 'lr': 1e-4},
            {'params': self.fc.parameters(), 'lr': 1e-3}
        ]
