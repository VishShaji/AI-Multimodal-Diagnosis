import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from processing.base_processor import BasePreprocessor
from typing import Dict, Any, List, Tuple, Optional
import yaml
from utils.logger import get_logger

class TextProcessor(BasePreprocessor):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.tokenizer = AutoTokenizer.from_pretrained(config['pretrained_model'])
        self.max_length = config['max_length']
        self.logger = get_logger(__name__)

    def process(self, text: str) -> Dict[str, torch.Tensor]:
        return self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

class TextDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], processor: TextProcessor):
        self.texts = texts
        self.labels = labels
        self.preprocessor = processor
        self.logger = get_logger(__name__)

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Optional[Dict[str, torch.Tensor]]:
        text = self.texts[idx]
        label = self.labels[idx]
        try:
            encoded_text = self.preprocessor.preprocess(text)
        except Exception as e:
            self.logger.error(f"Error processing text at index {idx}: {str(e)}")
            return None

        return {
            'input_ids': encoded_text['input_ids'].flatten(),
            'attention_mask': encoded_text['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

def load_text_data(texts: List[str], labels: List[int], config: Dict[str, Any], distributed: bool = False) -> DataLoader:
    processor = TextProcessor(config['processors']['text'])
    dataset = TextDataset(texts, labels, processor)
    
    sampler = torch.utils.data.distributed.DistributedSampler(dataset) if distributed else None
    
    return DataLoader(
        dataset,
        batch_size=config['data_loader']['batch_size'],
        shuffle=(sampler is None),
        num_workers=config['data_loader']['num_workers'],
        sampler=sampler
    )

if __name__ == "__main__":
    logger = get_logger(__name__)
    logger.info("Starting text processor")

    with open("model_config.yaml", "r") as config_file:
        config = yaml.safe_load(config_file)