from fastapi import FastAPI, HTTPException
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

app = FastAPI()

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

class PredictionRequest(BaseModel):
    data: str
    model_type: str

@app.post("/predict")
async def predict(request: PredictionRequest):
    try:
        # Validate model type
        if request.model_type not in prediction_pipeline.models:
            raise HTTPException(status_code=400, detail="Invalid model type")

        # Process the data according to the model type
        result = prediction_pipeline.predict({request.model_type: request.data})
        return {"prediction": result[request.model_type].tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Health check endpoint
@app.get("/health")
async def health_check():
    
    return {"status": "healthy"}
