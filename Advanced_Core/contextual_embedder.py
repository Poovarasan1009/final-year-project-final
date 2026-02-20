"""
Domain-adaptive embeddings for academic text
"""
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
import numpy as np
from typing import List, Dict

class DomainAdaptiveEmbedder:
    """Custom embedding model for academic text"""
    
    def __init__(self, model_name: str = "bert-base-uncased"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load pre-trained model
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Academic vocabulary enhancement
        self.academic_terms = {
            'algorithm', 'analysis', 'conclusion', 'data', 'experiment',
            'hypothesis', 'methodology', 'process', 'result', 'theory',
            'variable', 'constant', 'control', 'dependent', 'independent',
            'quantitative', 'qualitative', 'empirical', 'theoretical'
        }
        
        print(f"✓ DomainAdaptiveEmbedder loaded on {self.device}")
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts with domain adaptation"""
        if not isinstance(texts, list):
            texts = [texts]
        
        # Tokenize
        inputs = self.tokenizer(
            texts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)
        
        # Get embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Use mean pooling of last hidden state
        embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        
        return embeddings
    
    def similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts"""
        emb1 = self.encode(text1)
        emb2 = self.encode(text2)
        
        # Cosine similarity
        similarity = np.dot(emb1[0], emb2[0]) / (
            np.linalg.norm(emb1[0]) * np.linalg.norm(emb2[0])
        )
        
        return float(similarity)
    
    def batch_encode(self, texts: List[str], batch_size: int = 8) -> np.ndarray:
        """Encode multiple texts in batches"""
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = self.encode(batch)
            all_embeddings.append(batch_embeddings)
        
        return np.vstack(all_embeddings)