import torch
from transformers import  T5Tokenizer, T5ForSequenceClassification
from tqdm import tqdm


class RelevanceEvaluator:
    def __init__(self, model, tokenizer, device):
        
        self.device = device
        self.tokenizer, self.model = tokenizer, model


    def evaluate_relevance(self, question, contexts):
        """ 
        Generate relevance scores for each context in relation to the question

        parameter:
        question (str): The question from the user,
        contexts (list): list of strings representing retrieved contexts for the question.

        returns:
        List[str] : List of relevance scores
        """
        scores = []
        for context in tqdm(contexts, desc="Evaluating contexts"):
            if context=="":
                scores.append(-1.0)
                continue
            input_text = f"{question} [SEP] {context}"
            test = self.tokenizer(input_text, return_tensors="pt",padding="max_length",max_length=512).to(self.device)
            with torch.no_grad():  
              outputs = self.model(test["input_ids"].to(self.device),attention_mask=test["attention_mask"].to(self.device))
            
            scores.append(float(outputs["logits"].cpu()))
            
        # print(scores)
        return scores

    def flag_relevance(self, scores, threshold1, threshold2):

        """
        Convert scores into relevance flags based on thresholds

        parameters: 
        scores(list): list of score from evaluate_relevance
        threshold1, threshold2(float): threshold values to categorize into correct, incorrect, ambiguous 

        returns:
        list[int]: 2 for high relevance(correct), 1 for moderate relevance(ambiguous), 0 for low relevance(incorrect)
        """
        
        flags = []
        for score in scores:
            if score >= threshold1:
                flags.append(2)  # High relevance
            elif score >= threshold2:
                flags.append(1)  # Moderate relevance
            else:
                flags.append(0)  # Low relevance
        return flags

    def check_context_relevance(self, question, contexts, threshold1, threshold2):
        """
        Main method to evaluate and flag relevance for each context
        """
        
        scores = self.evaluate_relevance(question, contexts)
        flags = self.flag_relevance(scores, threshold1, threshold2)
        return flags



if __name__ == "__main__":
    
    tokenizer = T5Tokenizer.from_pretrained("909ahmed/model_t5")
    model = T5ForSequenceClassification.from_pretrained("909ahmed/model_t5", num_labels=1)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    evaluator = RelevanceEvaluator(model, tokenizer, device)


    # question = "What is the purpose of a URL?"
    # contexts = [
    #         "A URL (Uniform Resource Locator) is used to specify the address of a web page on the internet.",
    #         "HTML (HyperText Markup Language) is used to structure web pages.",
    #         "An IP address identifies a device on a network but differs from a URL in its usage.",
    #         "The internet was invented to facilitate global communication and data sharing."
    #     ]

    question=""
    contexts=[]

    threshold1 #= 0.59  # Threshold for high relevance
    threshold2 #= -0.95  # Threshold for moderate relevance

    relevance_flags = evaluator.check_context_relevance(question, contexts, threshold1, threshold2)
    print("Relevance Flags:", relevance_flags)

        
