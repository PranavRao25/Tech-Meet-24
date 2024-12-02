from transformers import AutoTokenizer, AutoModelForCausalLM

class AutoWrapper:
    def __init__(self, model_name_or_path):
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    def tokenize(self, text, **kwargs):
        return self.tokenizer.encode(text, return_tensors="pt")

    def __call__(self, text, **kwargs):
        if not isinstance(text, str):
            text = text.to_string()
        inputs = self.tokenize(text, **kwargs)
        output_ids = self.model.generate(inputs, max_new_tokens=100)
        return self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
    
    def to(self, device:str):
        self.model = self.model.to(device)
        return self