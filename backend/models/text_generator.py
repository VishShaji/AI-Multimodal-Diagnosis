from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class TextGenerator:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    @classmethod
    def from_pretrained(cls, model_name):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        return cls(model, tokenizer)

    def generate(self, input_text, max_length=512):
        sys_message = '''
        You are an AI Medical Assistant trained on a vast dataset of health information. Please be thorough and
        provide an informative answer. If you don't know the answer to a specific medical inquiry, advise seeking professional help.
        '''

        messages = [
            {"role": "system", "content": sys_message},
            {"role": "user", "content": input_text}
        ]

        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=max_length, 
                use_cache=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1
            )

        answer = self.tokenizer.batch_decode(outputs)[0].strip()
        return answer

    def __call__(self, input_text, **kwargs):
        return self.generate(input_text, **kwargs)
