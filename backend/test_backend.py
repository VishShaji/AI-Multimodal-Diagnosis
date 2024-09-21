import os
import yaml
import torch
from transformers import AutoModel
from models.cnn_model import CNNModel
from models.vit_model import ViTModel
from models.lstm_model import LSTMModel
from models.bert_classifier import BERTClassifier
from models.text_generator import TextGenerator
from utils.logger import setup_logger

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
        # Initialize CNN model
        models['cnn'] = CNNModel.from_pretrained(config['cnn']['pretrained_model'],
                                                 num_classes=config['cnn']['num_classes'])
        
        # Initialize ViT model
        vit_model = AutoModel.from_pretrained(config['vit']['pretrained_model'])
        models['vit'] = ViTModel(vit_model, num_classes=config['vit']['num_classes'])
        
        # Initialize LSTM model
        models['lstm'] = LSTMModel.from_pretrained(**config['lstm'])
        
        # Initialize BERT model
        bert_model = AutoModel.from_pretrained(config['bert']['pretrained_model'])
        models['bert'] = BERTClassifier(bert_model, **config['bert'])
        
        # Initialize text generation model
        models['text_gen'] = TextGenerator.from_pretrained(config['text_gen']['pretrained_model'])
    
    except KeyError as e:
        raise ValueError(f"Missing configuration for model: {str(e)}")
    
    return models

def predict(models):
    # Simulating input data for predictions; replace with real inputs in practice.
    cnn_input = torch.rand((1, 3, 224, 224))  # Random tensor simulating an image input for CNN
    vit_input = torch.rand((1, 3, 224, 224))  # Random tensor simulating an image input for ViT
    lstm_input = torch.rand((1, 10, 30))     # Random tensor simulating time series input for LSTM
    bert_input = torch.randint(0, 100, (1, 30))  # Random tensor simulating tokenized text input for BERT
    
    # Predictions
    cnn_prediction = models['cnn'].predict(cnn_input)
    vit_prediction = models['vit'].predict(vit_input)
    lstm_prediction = models['lstm'].predict(lstm_input)
    bert_prediction = models['bert'].predict(bert_input)
    
    print(f"CNN Prediction: {cnn_prediction}")
    print(f"ViT Prediction: {vit_prediction}")
    print(f"LSTM Prediction: {lstm_prediction}")
    print(f"BERT Prediction: {bert_prediction}")

def main():
    logger = setup_logger('test_backend_logger', 'logs/test_backend.log')
    
    try:
        # Load configuration
        config = load_config('config/model_config.yaml')
        
        # Initialize models
        models = initialize_models(config)
        
        # Test model predictions
        print("Testing model predictions with random inputs...")
        predict(models)
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()
