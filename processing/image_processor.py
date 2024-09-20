import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from typing import Dict, Any, List, Tuple, Optional
import yaml
from processing.base_processor import BaseProcessor
from logger import get_logger

class ImageProcessor(BaseProcessor):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.transform = transforms.Compose([
            transforms.Resize(tuple(config['resize'])),
            transforms.ToTensor(),
            transforms.Normalize(mean=config['normalize_mean'], std=config['normalize_std'])
        ])
        self.logger = get_logger(__name__)

    def process(self, image: Image.Image) -> torch.Tensor:
        self.logger.debug(f"Processing image of size {image.size}")
        return self.transform(image)

class ImageDataset(Dataset):
    def __init__(self, data_dir: str, processor: ImageProcessor):
        self.data_dir = data_dir
        self.processor = processor
        self.image_files = [f for f in os.listdir(data_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        self.label_encoder = self._create_label_encoder()
        self.logger = get_logger(__name__)
        self.logger.info(f"Initialized ImageDataset with {len(self.image_files)} images")

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Optional[Tuple[torch.Tensor, int]]:
        img_name = os.path.join(self.data_dir, self.image_files[idx])
        try:
            with Image.open(img_name) as image:
                image = image.convert('RGB')
                image = self.processor.process(image)
        except Exception as e:
            self.logger.error(f"Error processing image {img_name}: {str(e)}")
            return None

        label = self.image_files[idx].split('_')[0]
        label_id = self.label_encoder[label]
        
        return image, label_id

    def _create_label_encoder(self) -> Dict[str, int]:
        unique_labels = set(f.split('_')[0] for f in self.image_files)
        return {label: idx for idx, label in enumerate(sorted(unique_labels))}

def load_image_data(data_dir: str, config: Dict[str, Any], distributed: bool = False) -> DataLoader:
    logger = get_logger(__name__)
    processor = ImageProcessor(config['processors']['image'])
    dataset = ImageDataset(data_dir, processor)
    
    sampler = torch.utils.data.distributed.DistributedSampler(dataset) if distributed else None
    
    dataloader = DataLoader(
        dataset,
        batch_size=config['data_loader']['batch_size'],
        shuffle=(sampler is None),
        num_workers=config['data_loader']['num_workers'],
        sampler=sampler
    )
    
    logger.info(f"Created DataLoader with {len(dataset)} samples, batch size {config['data_loader']['batch_size']}")
    return dataloader

if __name__ == "__main__":
    logger = get_logger(__name__)
    logger.info("Starting image processor")
    
    with open("model_config.yaml", "r") as config_file:
        config = yaml.safe_load(config_file)