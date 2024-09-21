from abc import ABC, abstractmethod

class BaseProcessor(ABC):
    @abstractmethod
    def __init__(self, config):
        self.config = config

    @abstractmethod
    def preprocess(self, data):
        pass

    def postprocess(self, data):
        return data