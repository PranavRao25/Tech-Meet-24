from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

class Thresholder:
    def __init__(self, model_name="geektech/flan-t5-large-lora-qa-gpt4"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Load the model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model = self.model.to(self.device)
        self.model.eval()

    def generate_relevance_prompt(self, question, document):
        """Generate a prompt for the relevance evaluation task."""
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

    def check_relevance_for_all_contexts(self, question, contexts):
        """
        Evaluate relevance scores for multiple contexts.

        Args:
            question (str): The user-provided question.
            contexts (list): A list of document contexts.

        Returns:
            list: A list of relevance scores.
        """
        results = []
        for context in contexts:
            prompt = self.generate_relevance_prompt(question, context)
            inputs = self.tokenizer(prompt, return_tensors="pt", padding="max_length", max_length=512).to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(inputs.input_ids, max_length=10)

            # Decode the model's output and parse the relevance score
            relevance_score_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

            try:
                # Convert the text response to a numerical score
                relevance_score = float(relevance_score_text)
            except ValueError:
                # Handle cases where the model outputs textual classifications
                if relevance_score_text == "Highly Relevant":
                    relevance_score = 1.0
                elif relevance_score_text == "Partially Relevant":
                    relevance_score = 0.0
                elif relevance_score_text in ["Not Relevant or Contradictory", "Not Relevant"]:
                    relevance_score = -1.0
                else:
                    relevance_score = 0.0  # Default for unexpected responses

            results.append(relevance_score)

        return results
