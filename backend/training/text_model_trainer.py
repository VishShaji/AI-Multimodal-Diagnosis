import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from training.base_trainer import BaseTrainer

class TextModelTrainer(BaseTrainer):
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def train(self, model, data_loader, epochs):
        model.to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.get_fine_tuning_parameters(), lr=self.config['learning_rate'])

        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            for batch in tqdm(data_loader):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                optimizer.zero_grad()
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            
            print(f'Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(data_loader)}')