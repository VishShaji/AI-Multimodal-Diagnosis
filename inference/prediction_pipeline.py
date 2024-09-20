import torch

class PredictionPipeline:
    def __init__(self, models, processors):
        self.models = models
        self.processors = processors
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def predict(self, data):
        results = {}
        for model_name, model in self.models.items():
            model.to(self.device)
            model.eval()

            try:
                processed_data = self.processors[model_name].process(data)
            except KeyError:
                raise ValueError(f"No processor found for model: {model_name}")

            # Make predictions based on the model type
            with torch.no_grad():
                if isinstance(processed_data, dict):  # For BERT input
                    input_ids = processed_data['input_ids'].to(self.device)
                    attention_mask = processed_data['attention_mask'].to(self.device)
                    output = model(input_ids=input_ids, attention_mask=attention_mask)
                else:  # For other model types
                    processed_data = processed_data.to(self.device)
                    output = model(processed_data)

            # Store the output
            results[model_name] = output.cpu().numpy()
        
        return results