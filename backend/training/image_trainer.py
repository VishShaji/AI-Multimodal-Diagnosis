import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from training.base_trainer import BaseTrainer

class ImageModelTrainer(BaseTrainer):
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
            for inputs, labels in tqdm(data_loader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            
            print(f'Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(data_loader)}')