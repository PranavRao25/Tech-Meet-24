from transformers import RobertaTokenizer, RobertaForSequenceClassification
import os
import toml
config=toml.load('config.toml')
HF_TOKEN=config['HF_TOKEN']
os.environ['HF_TOKEN']= HF_TOKEN
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer
model_gib = AutoModelForSequenceClassification.from_pretrained("madhurjindal/autonlp-Gibberish-Detector-492513457")

tokenizer_gib = AutoTokenizer.from_pretrained("madhurjindal/autonlp-Gibberish-Detector-492513457")

# Load the model and tokenizer
model_name = "cardiffnlp/twitter-roberta-base-offensive"
tokenizer = RobertaTokenizer.from_pretrained(model_name)
model = RobertaForSequenceClassification.from_pretrained(model_name)

# Ensure the model uses the CPU (default behavior)
device = torch.device('cpu')  # Set device to CPU

# Move the model to CPU (though it's already on CPU by default)
model.to(device)

def query(query):
    if query.strip() == "":
        return "I'm sorry, Could you please provide some input so I can assist you?"
    
    inputs_gib = tokenizer_gib(query, return_tensors="pt")

    outputs_gib = model_gib(**inputs_gib)

    predictions_gib = F.softmax(outputs_gib.logits, dim=-1)
    predicted_class_gib = torch.argmax(predictions_gib, dim=-1).item()
    labels_gib = model_gib.config.id2label  # Access the labels defined in the model config
    predicted_label_gib = labels_gib[predicted_class_gib]
    score_gib = predictions_gib[0][predicted_class_gib].item()
    if (predicted_label_gib.lower() == "noise" and score_gib > 0.75) or (predicted_label_gib.lower() == "word salad" and score_gib > 0.9):
        return "I'm sorry, I didn't quite understand that. Could you please rephrase or clarify your question?"


    inputs = tokenizer(query, return_tensors="pt", truncation=True, padding=True, max_length=512)
    
    # Move input tensors to CPU (this is optional since it's CPU by default)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    
    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get the predicted label (offensive or not)
    predictions = F.softmax(outputs.logits, dim=-1)
    predicted_class = torch.argmax(predictions, dim=-1).item()
    labels = model.config.id2label  # Access the labels defined in the model config
    predicted_label = labels[predicted_class]
    score = predictions[0][predicted_class].item()
    
    # print(f"Moderation result for query '{query}': {predicted_label}, score: {score}")  # Debugging
    
    # Only raise an error if the label is "offensive"
    if predicted_label.lower() == "offensive" and score > 0.5:
        return "I'm sorry, but I cannot engage in that topic. Let's keep the conversation respectful and appropriate."
    return None
if __name__=='__main__':
    output = query(input("Enter"))
    print(output)
