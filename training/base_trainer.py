from abc import ABC, abstractmethod

class BaseTrainer(ABC):
    @abstractmethod
    def train(self, model, data_loader, epochs):
        pass