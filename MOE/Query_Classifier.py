import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
import torch.nn.functional as F
import spacy
from typing import Dict


class ExpertNetwork(nn.Module):
    def __init__(self, hidden_size: int, expert_hidden_size: int = 256):
        super().__init__()
        self.expert = nn.Sequential(
            nn.Linear(hidden_size, expert_hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(expert_hidden_size, 3)  # 3 classes: simple, intermediate, complex
        )

    def forward(self, x):
        return self.expert(x)


class QueryComplexityMoE(nn.Module):
    def __init__(self, model_name: str = "microsoft/deberta-v3-small", num_experts: int = 8):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.encoder.config.hidden_size
        self.experts = nn.ModuleList([ExpertNetwork(self.hidden_size) for _ in range(num_experts)])
        self.gate = nn.Sequential(
            nn.Linear(self.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_experts)
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = 512
        self.nlp = spacy.load("en_core_web_sm")

    def calculate_complexity_features(self, query: str) -> Dict[str, float]:
        query = query.lower()
        features = {
            "basic_complexity": 0.3 * sum(1 for indicator in ["select", "from", "where"] if indicator in query),
            "join_complexity": 3.0 * sum(1 for indicator in ["join", "inner join", "left join", "right join"] if indicator in query),
            "aggregation_complexity": 2.5 * sum(1 for indicator in ["group by", "having", "sum(", "avg(", "count("] if indicator in query),
            "temporal_complexity": 1.5 * sum(1 for indicator in ["between", "date", "period", "range", "interval"] if indicator in query),
            "nested_complexity": 3.0 * sum(1 for indicator in ["in (select", "exists", "not exists", "subquery"] if indicator in query),
            "window_complexity": 3.0 * sum(1 for indicator in ["over (", "partition by", "rank()", "dense_rank"] if indicator in query),
            "set_complexity": 2.0 * sum(1 for indicator in ["union", "intersect", "except", "minus"] if indicator in query),
            "distinct_complexity": 1.0 * query.count("distinct"),
            "case_complexity": 2.0 * query.count("case when"),
            "function_complexity": 1.5 * sum(1 for indicator in ["concat", "substring", "trim", "coalesce"] if indicator in query),
            "entity_complexity": 2.0 * len(self.nlp(query).ents)
        }
        return features

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, complexity_bias: float = 0.0) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        gate_logits = self.gate(pooled_output)
        gate_weights = F.softmax(gate_logits, dim=-1)

        expert_outputs = torch.stack([expert(pooled_output) for expert in self.experts])
        gate_weights = gate_weights.unsqueeze(-1)
        expert_outputs = expert_outputs.permute(1, 0, 2)

        bias = torch.tensor([[complexity_bias * -2, 0.0, complexity_bias * 2]]).to(expert_outputs.device)
        final_output = torch.sum(expert_outputs * gate_weights, dim=1) + bias

        return final_output

    def classify_query(self, query: str) -> str:
        features = self.calculate_complexity_features(query)
        complexity_score = sum(features.values())

        if complexity_score <= 0.5:
            complexity_bias = -1.5
        elif complexity_score <= 2.0:
            complexity_bias = -1.0
        elif complexity_score <= 4.0:
            complexity_bias = 0.5
        else:
            complexity_bias = 1.5

        tokenizer_output = self.tokenizer(query, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length)
        inputs = {
            'input_ids': tokenizer_output['input_ids'],
            'attention_mask': tokenizer_output['attention_mask']
        }

        with torch.no_grad():
            logits = self.forward(**inputs, complexity_bias=complexity_bias)
            probs = F.softmax(logits, dim=-1)
            predicted_class = torch.argmax(probs[0]).item()

        class_mapping = {0: "simple", 1: "intermediate", 2: "complex"}
        return class_mapping[predicted_class]


class QueryClassifier:
    def __init__(self, model_name: str = "microsoft/deberta-v3-small", num_experts: int = 8):
        self.model = QueryComplexityMoE(model_name=model_name, num_experts=num_experts)
        self.model.eval()

    def classify(self, query: str) -> str:
        """
        Classify the given query into one of the following categories:
        - "simple"
        - "intermediate"
        - "complex"

        Args:
            query (str): The input query string.

        Returns:
            str: The predicted class of the query.
        """
        return self.model.classify_query(query)