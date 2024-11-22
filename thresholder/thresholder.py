from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("geektech/flan-t5-large-lora-qa-gpt4")
model = AutoModelForSeq2SeqLM.from_pretrained("geektech/flan-t5-large-lora-qa-gpt4")
model=model.to(device)
model.eval()

# Define the prompt template
def generate_relevance_prompt(question, document):
    return (
        f'''You are an evaluator tasked with assessing how relevant a document is to a user’s question. Your evaluation should consider the semantic alignment between the question and the document.

        Instructions:
        1. **Highly Relevant (Score: 0.5 to 1.0)**:
          - The document directly answers the question or fully aligns with its meaning.
          - Examples include direct answers, precise explanations, or highly related content.

        2. **Partially Relevant (Score: 0.0 to 0.5)**:
          - The document is somewhat related but misses key details or only vaguely addresses the question.

        3. **Not Relevant or Contradictory (Score: -1.0 to 0.0)**:
          - The document is unrelated, off-topic, or contradicts the question.

        Your output should only be a floating-point number between -1.0 and 1.0, based on the relevance criteria.

        ### Question:
        {question}

        ### Document:
        {document}

        ### Relevance Score:
        '''
    )

# Function to check relevance for multiple contexts
def check_relevance_for_all_contexts(question, contexts):
    results = []
    for context in contexts:
        prompt = generate_relevance_prompt(question, context)
        inputs = tokenizer(prompt, return_tensors="pt", padding="max_length", max_length=512).to(device)

        with torch.no_grad():
            outputs = model.generate(inputs.input_ids, max_length=10)

        # Decode the model's output and parse the relevance score
        relevance_score_text = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        # print(f"Raw Model Output: {relevance_score_text}")

        try:
            # Convert the text response to a numerical score
            relevance_score = float(relevance_score_text)
        except ValueError:
            # Handle cases where the model outputs textual classifications
            if relevance_score_text == "Highly Relevant":
                relevance_score = 1.0
            elif relevance_score_text == "Partially Relevant":
                relevance_score = 0.0
            elif relevance_score_text == "Not Relevant or Contradictory" or relevance_score_text == "Not Relevant" :
                relevance_score = -1.0
            else:
                relevance_score = 0.0  # Default for unexpected responses

        # Determine if the context is relevant based on the threshold
        is_relevant = relevance_score >= 0 # threshold
        results.append(relevance_score)

    return results

question=""
contexts=[]

results = check_relevance_for_all_contexts(question, contexts)

for relevance_score in results:
    print(f"\nRelevance Score: {relevance_score}")
