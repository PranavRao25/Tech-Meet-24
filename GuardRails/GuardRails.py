import requests
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch

# Load the model and tokenizer
model_name = "cardiffnlp/twitter-roberta-base-offensive"
tokenizer = RobertaTokenizer.from_pretrained(model_name)
model = RobertaForSequenceClassification.from_pretrained(model_name)

# Ensure the model uses the CPU (default behavior)
device = torch.device('cpu')  # Set device to CPU

# Move the model to CPU (though it's already on CPU by default)
model.to(device)

API_URL = "https://api-inference.huggingface.co/models/madhurjindal/autonlp-Gibberish-Detector-492513457"
headers = {"Authorization": "Bearer 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"}

def query(query):
    if query.strip() == "":
        return "I'm sorry, please give valid query"
    payload={"inputs": query}
    for i in range(3):
        try:
            response = requests.post(API_URL, headers=headers, json=payload)
            response_json= response.json()
            # print(response_json[0])
            # find the scores of label noise and label word_salad
            noise_score=0
            salad_score=0
            print(response_json)
            for i in response_json[0]:
                if i['label']=='noise':
                    noise_score=i['score']
                if i['label']=='word salad':
                    salad_score=i['score']
            if noise_score > 0.75 or salad_score>0.9:
                return "I'm sorry, I didn't quite understand that. Could you please rephrase or clarify your question?"
            break
        except:
            if i == 2:
                return "It seems like there was internal error" # when API POST Request
    
    inputs = tokenizer(query, return_tensors="pt", truncation=True, padding=True, max_length=512)
    
    # Move input tensors to CPU (this is optional since it's CPU by default)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    
    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get the predicted label (offensive or not)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predicted_class = torch.argmax(predictions, dim=-1).item()
    labels = model.config.id2label  # Access the labels defined in the model config
    predicted_label = labels[predicted_class]
    score = predictions[0][predicted_class].item()
    
    # print(f"Moderation result for query '{query}': {predicted_label}, score: {score}")  # Debugging
    
    # Only raise an error if the label is "offensive"
    if predicted_label.lower() == "offensive" and score > 0.5:
        return "I'm sorry, but I cannot engage in that topic. Let's keep the conversation respectful and appropriate."

if __name__=='__main__':
    output = query(input("Enter"))




# Function to check if a query contains offensive content


















# from guardrails.hub import GibberishText
# from guardrails.hub import NSFWText
# from guardrails import Guard
# def validate_query(query)->str:
#     """
#     Return None if the input is valid, otherwise return the error message.
#     """
#     # Use the Guard with the validator
#     Gibberish_guard = Guard().use(
#         GibberishText, threshold=0.5, validation_method="sentence", on_fail="exception"
#     )
#     try:
#         # Test failing response
#         Gibberish_guard.validate(query)
#     except Exception as e:
#         return "I'm sorry, I didn't quite understand that. Could you please rephrase or clarify your question?"
    
#     NSFW_guard = Guard().use(
#     NSFWText, threshold=0.8, validation_method="sentence", on_fail="exception")

#     try:
#         # Test failing response
#         # Gibberish_guard.validate(query)
#         NSFW_guard.validate(query)
#     except Exception as e:
#         return "I'm sorry, but I cannot engage in that topic. Let's keep the conversation respectful and appropriate."
#     return None
# if __name__ == '__main__':
    
#     print(validate_query(input("Enter a query: ")))