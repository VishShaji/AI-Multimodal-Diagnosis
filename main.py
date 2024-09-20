import os
import yaml
import torch
from transformers import AutoModel
from models.cnn_model import CNNModel
from models.vit_model import ViTModel
from models.lstm_model import LSTMModel
from models.bert_classifier import BERTClassifier
from training.image_trainer import ImageModelTrainer
from training.time_series_trainer import TimeSeriesModelTrainer
from training.text_model_trainer import TextModelTrainer
from processing.image_processor import load_image_data
from processing.text_processor import load_text_data
from processing.text_processor import load_time_series_data
from utils.logger import setup_logger
from utils.error_handling import handle_model_not_found, handle_invalid_input

def load_config(config_path):
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML configuration: {str(e)}")
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

def initialize_models(config):
    models = {}
    try:

        models['cnn'] = CNNModel.from_pretrained(config['cnn']['pretrained_model'], 
                                             num_classes=config['cnn']['num_classes'])

        vit_model = AutoModel.from_pretrained(config['vit']['pretrained_model'])
        models['vit'] = ViTModel(vit_model, num_classes=config['vit']['num_classes'])

        models['lstm'] = LSTMModel.from_pretrained(**config['lstm'])

        bert_model = AutoModel.from_pretrained(config['bert']['pretrained_model'])
        models['bert'] = BERTClassifier(bert_model, **config['bert'])


    except KeyError as e:
        raise ValueError(f"Missing configuration for model: {str(e)}")
    return models

def train_models(models, trainers, data_loaders, config):
    for model_name, model in models.items():
        trainer = trainers[model_name]
        data_loader = data_loaders[model_name]
        epochs = config[f'{model_name}_model']['epochs']
        
        try:
            trainer.train(model, data_loader, epochs=epochs)
        except Exception as e:
            raise RuntimeError(f"Error training {model_name} model: {str(e)}")

def save_models(models, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    for model_name, model in models.items():
        try:
            torch.save(model.state_dict(), f'{save_dir}/{model_name}_model.pth')
        except Exception as e:
            raise IOError(f"Error saving {model_name} model: {str(e)}")

def main():
    logger = setup_logger('main_logger', 'logs/main.log')
    
    try:
        config = load_config('config/model_config.yaml')
        
        data_loaders = {
            'cnn': load_image_data('path/to/image/data', distributed=True),
            'vit': load_image_data('path/to/image/data', distributed=True),
            'lstm': load_time_series_data('path/to/time_series/data', distributed=True),
            'bert': load_text_data('path/to/text/data', distributed=True)
        }
        
        models = initialize_models(config)
        
        trainers = {
            'cnn': ImageModelTrainer(config['cnn_model']),
            'vit': ImageModelTrainer(config['vit_model']),
            'lstm': TimeSeriesModelTrainer(config['lstm_model']),
            'bert': TextModelTrainer(config['bert_model'])
        }
        
        logger.info("Starting model training...")
        train_models(models, trainers, data_loaders, config)
        logger.info("Model training completed.")
        
        save_models(models, 'saved_models')
        logger.info("Models saved successfully.")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()