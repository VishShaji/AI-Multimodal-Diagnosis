import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from processing.base_processor import BasePreprocessor
from typing import Dict, Any, List, Tuple
import yaml
from utils.logger import get_logger

class TimeSeriesProcessor(BasePreprocessor):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.sequence_length = config['sequence_length']
        self.logger = get_logger(__name__)

    def process(self, data: np.ndarray) -> np.ndarray:
        normalized_data = (data - np.mean(data)) / np.std(data)
        # Create sequences
        sequences = []
        for i in range(len(normalized_data) - self.sequence_length):
            sequence = normalized_data[i:i+self.sequence_length]
            sequences.append(sequence)
        
        return np.array(sequences)

class TimeSeriesDataset(Dataset):
    def __init__(self, data: np.ndarray, processor: TimeSeriesProcessor):
        self.data = data
        self.preprocessor = processor
        self.sequences = self.preprocessor.process(data)
        self.logger = get_logger(__name__)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sequence = self.sequences[idx]
        target = self.data[idx + self.preprocessor.sequence_length]
        return torch.FloatTensor(sequence), torch.FloatTensor([target])

def load_time_series_data(data: np.ndarray, config: Dict[str, Any], distributed: bool = False) -> DataLoader:
    preprocessor = TimeSeriesProcessor(config['processors']['time_series'])
    dataset = TimeSeriesDataset(data, preprocessor)
    
    sampler = torch.utils.data.distributed.DistributedSampler(dataset) if distributed else None
    
    return DataLoader(
        dataset,
        batch_size=config['data_loader']['batch_size'],
        shuffle=(sampler is None),
        num_workers=config['data_loader']['num_workers'],
        sampler=sampler
    )
