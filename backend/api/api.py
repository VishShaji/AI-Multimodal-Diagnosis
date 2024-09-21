from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from inference.prediction_pipeline import PredictionPipeline
from models.cnn_model import CNNModel
from models.bert_classifier import BERTClassifier
from models.lstm_model import LSTMModel  
from models.vit_model import ViTModel
from processing.image_processor import ImageProcessor
from processing.text_processor import TextProcessor
from processing.time_series_processor import TimeSeriesProcessor
import yaml
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)

app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Origin (TO BE ADJUSTED DURING PRODUCTION)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load configuration
with open('config/model_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Initialize models
cnn_model = CNNModel(num_classes=config['models']['cnn']['num_classes'], 
                     weight_path=config['models']['cnn']['weight_path'])
bert_model = BERTClassifier(pretrained_model=config['models']['bert']['pretrained_model'], 
                            num_classes=config['models']['bert']['num_classes'])
lstm_model = LSTMModel(pretrained_model_name=config['models']['lstm']['pretrained_embeddings'], 
                        hidden_size=config['models']['lstm']['hidden_size'], 
                        num_layers=config['models']['lstm']['num_layers'], 
                        output_size=config['models']['lstm']['output_size'])
vit_model = ViTModel(pretrained_model=config['models']['vit']['pretrained_model'], 
                     num_classes=config['models']['vit']['num_classes'])

# Initialize data processors
image_processor = ImageProcessor(config['processors']['image_processor'])
text_processor = TextProcessor(config['processors']['text_processor'])
time_series_processor = TimeSeriesProcessor(config['processors']['time_series_processor'])

# Initialize prediction pipeline
prediction_pipeline = PredictionPipeline(
    models={
        'cnn': cnn_model,
        'bert': bert_model,
        'lstm': lstm_model,
        'vit': vit_model
    },
    processors={
        'cnn': image_processor,
        'bert': text_processor,
        'lstm': time_series_processor
    }
)

@app.post("/predict")
async def predict(
    model_type: str = Form(...),  # Model type sent as a form field
    cnn_file: UploadFile = File(None),  # CNN expects an image file
    bert_text: str = Form(None),  # BERT expects text input
    lstm_file: UploadFile = File(None),  # LSTM expects a CSV file
    vit_file: UploadFile = File(None)  # ViT expects an image file
):
    try:
        # Validate model type
        if model_type not in prediction_pipeline.models:
            raise HTTPException(status_code=400, detail="Invalid model type")

        data = None

        # Process CNN data
        if model_type == "cnn" and cnn_file:
            content = await cnn_file.read() 
            image = image_processor.process_image(content)
            data = np.array(image)

        # Process BERT text data
        elif model_type == "bert" and bert_text:
            data = text_processor.process_text(bert_text)

        # Process LSTM time series data
        elif model_type == "lstm" and lstm_file:
            content = await lstm_file.read()
            csv_data = time_series_processor.process_time_series(content)
            data = np.array(csv_data)

        # Process ViT data
        elif model_type == "vit" and vit_file:
            content = await vit_file.read()
            image = image_processor.process_image(content)
            data = np.array(image)

        if data is None:
            raise HTTPException(status_code=400, detail="No valid data provided")

        # Run prediction
        result = prediction_pipeline.predict({model_type: data})

        return {"prediction": result[model_type].tolist()}

    except Exception as e:
        logging.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}
