# import requests

# API_URL = "https://api-inference.huggingface.co/models/deepset/roberta-base-squad2"
# headers = {"Authorization": "Bearer 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"}

# def query(payload):
# 	response = requests.post(API_URL, headers=headers, json=payload)
# 	return response.json()
	
# output = query({
# 	"inputs": {
# 	"question": "What is my name?",
# 	"context": "My name is Clara and I live in Berkeley."
# },
# })
# print(output)









import requests

API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1"
headers = {"Authorization": "Bearer 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"}

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()
	
output = query({
	"inputs": {"question":"What is the capital of France? "},
})
my_query="What is the capital of France?"

print(output)


# from huggingface_hub import InferenceClient
# print("starting")
# client = InferenceClient(api_key="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")
# print("client created")
# messages = [
# 	{
# 		"role": "user",
# 		"content": "What is the capital of France?"
# 	}
# ]

# completion = client.chat.completions.create(
#     model="mistralai/Mistral-7B-Instruct-v0.1", 
# 	messages=messages, 
# 	max_tokens=500
# )

# print(completion.choices[0].message)






