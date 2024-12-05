import requests

API_URL = "https://api-inference.huggingface.co/models/madhurjindal/autonlp-Gibberish-Detector-492513457"
headers = {"Authorization": "Bearer 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"}

def query(query):
    if query.strip() == "":
        return "I'm sorry, please give valid query"
    payload={"inputs": query}
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
    else:
        return None
if __name__=='__main__':
    output = query(input("Enter"))























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