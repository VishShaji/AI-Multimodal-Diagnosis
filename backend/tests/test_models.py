import unittest
import torch
import torch.nn as nn
import yaml
from models.cnn_model import CNNModel
from models.lstm_model import LSTMModel
from models.vit_model import ViTModel
from models.bert_model import BERTClassifier

# Load configuration
with open('model_config.yaml', 'r') as file:
    config = yaml.safe_load(file)

class TestModels(unittest.TestCase):
    def setUp(self):
        self.batch_size = config['data_loader']['batch_size']
        self.seq_length = config['processors']['text']['max_length']
        
        # Initialize models
        self.cnn_model = CNNModel(
            num_classes=config['models']['cnn']['num_classes'],
            pretrained_model=config['models']['cnn']['pretrained_model']
        )
        self.lstm_model = LSTMModel(
            pretrained_model=config['models']['lstm']['pretrained_model'],
            input_size=config['models']['lstm']['input_size'],
            hidden_size=config['models']['lstm']['hidden_size'],
            num_layers=config['models']['lstm']['num_layers'],
            output_size=config['models']['lstm']['output_size'],
            dropout_rate=config['models']['lstm']['dropout_rate']
        )
        self.vit_model = ViTModel(
            pretrained_model=config['models']['vit']['pretrained_model'],
            num_classes=config['models']['vit']['num_classes']
        )
        self.bert_model = BERTClassifier(
            pretrained_model=config['models']['bert']['pretrained_model'],
            num_classes=config['models']['bert']['num_classes'],
            dropout_rate=config['models']['bert']['dropout_rate']
        )
        
        # Prepare input tensors
        self.image_tensor = torch.randn(self.batch_size, 3, *config['processors']['image']['resize'])
        self.text_tensor = torch.randint(0, 1000, (self.batch_size, self.seq_length))
        self.attention_mask = torch.ones((self.batch_size, self.seq_length), dtype=torch.long)

    def test_model_structure(self):
        models = [self.cnn_model, self.lstm_model, self.vit_model, self.bert_model]
        for model in models:
            self.assertIsInstance(model, nn.Module)

    def test_forward_pass(self):
        cnn_output = self.cnn_model(self.image_tensor)
        self.assertEqual(cnn_output.shape, torch.Size([self.batch_size, config['models']['cnn']['num_classes']]))
        
        lstm_output = self.lstm_model(self.text_tensor, self.attention_mask)
        self.assertEqual(lstm_output.shape, torch.Size([self.batch_size, config['models']['lstm']['output_size']]))
        
        vit_output = self.vit_model(self.image_tensor)
        self.assertEqual(vit_output.shape, torch.Size([self.batch_size, config['models']['vit']['num_classes']]))
        
        bert_output = self.bert_model(self.text_tensor, self.attention_mask)
        self.assertEqual(bert_output.shape, torch.Size([self.batch_size, config['models']['bert']['num_classes']]))

    def test_prediction(self):
        models = [
            (self.cnn_model, self.image_tensor),
            (self.lstm_model, (self.text_tensor, self.attention_mask)),
            (self.vit_model, self.image_tensor),
            (self.bert_model, (self.text_tensor, self.attention_mask))
        ]
        
        for model, inputs in models:
            if isinstance(inputs, tuple):
                prediction = model.predict(*inputs)
            else:
                prediction = model.predict(inputs)
            self.assertEqual(prediction.shape, torch.Size([self.batch_size]))
            self.assertTrue(torch.all((prediction >= 0) & (prediction < max(model.num_classes, 2))))

    def test_fine_tuning_parameters(self):
        models = [self.cnn_model, self.lstm_model, self.vit_model, self.bert_model]
        for model in models:
            params = model.get_fine_tuning_parameters()
            self.assertIsInstance(params, list)
            self.assertGreater(len(params), 0)
            for param_group in params:
                self.assertIn('params', param_group)
                self.assertIn('lr', param_group)

    def test_model_on_device(self):
        models = [self.cnn_model, self.lstm_model, self.vit_model, self.bert_model]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for model in models:
            model.to(device)
            self.assertEqual(next(model.parameters()).device, device)

if __name__ == '__main__':
    unittest.main()