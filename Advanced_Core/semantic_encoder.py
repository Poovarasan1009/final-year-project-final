import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

class SemanticEncoder:
    """
    Sentence-BERT for semantic similarity
    Loss Function: Contrastive loss with cosine similarity
    L = -log(exp(sim(q, p)/τ) / Σ exp(sim(q, n)/τ))
    """
    
    def __init__(self, model_name='all-mpnet-base-v2', device='cpu'):
        self.device = device
        self.model_name = model_name
        
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name).to(device)
            self.model.eval()
            self.dimension = self.model.get_sentence_embedding_dimension()
            print(f"✓ Loaded SBERT model: {model_name}, dimension: {self.dimension}")
        except Exception as e:
            print(f"⚠ Could not load SBERT: {e}")
            self.model = None
            self.dimension = 768
            self._initialize_fallback()
    
    def _initialize_fallback(self):
        """Fallback TF-IDF based encoder"""
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
        self.vocab_loaded = False
    
    def encode(self, texts: List[str], convert_to_tensor=True) -> torch.Tensor:
        """
        Encode sentences to embeddings
        Formula: E(sentence) = mean_pooling(BERT_token_embeddings)
        """
        if self.model is not None:
            # Use SBERT
            embeddings = self.model.encode(texts, 
                                         convert_to_tensor=convert_to_tensor,
                                         device=self.device,
                                         show_progress_bar=False)
            return embeddings
        else:
            # Fallback: TF-IDF
            if not self.vocab_loaded:
                # Fit on some dummy data
                dummy_texts = ["machine learning", "artificial intelligence", 
                              "data science", "natural language processing"]
                self.tfidf.fit(dummy_texts)
                self.vocab_loaded = True
            
            tfidf_vectors = self.tfidf.transform(texts).toarray()
            if convert_to_tensor:
                return torch.tensor(tfidf_vectors, dtype=torch.float32)
            return tfidf_vectors
    
    def cosine_similarity(self, embedding1: torch.Tensor, 
                         embedding2: torch.Tensor) -> float:
        """
        Calculate cosine similarity
        Formula: cos_sim(A,B) = (A·B) / (||A|| * ||B||)
        """
        if embedding1.dim() == 1:
            embedding1 = embedding1.unsqueeze(0)
        if embedding2.dim() == 1:
            embedding2 = embedding2.unsqueeze(0)
        
        # Normalize vectors
        embedding1_norm = F.normalize(embedding1, p=2, dim=1)
        embedding2_norm = F.normalize(embedding2, p=2, dim=1)
        
        # Calculate cosine similarity
        similarity = torch.mm(embedding1_norm, embedding2_norm.T)
        
        return similarity.item()
    
    def semantic_score(self, ideal_answer: str, student_answer: str) -> Dict:
        """
        Calculate semantic similarity between two answers
        Returns: score, details, confidence
        """
        # Encode both answers
        embeddings = self.encode([ideal_answer, student_answer], convert_to_tensor=True)
        ideal_emb = embeddings[0]
        student_emb = embeddings[1]
        
        # Calculate cosine similarity
        cos_sim = self.cosine_similarity(ideal_emb, student_emb)
        
        # Convert from [-1, 1] to [0, 1] range
        normalized_score = (cos_sim + 1) / 2
        
        # Calculate confidence based on embedding quality
        confidence = self._calculate_confidence(ideal_emb, student_emb)
        
        # Additional metrics
        jaccard = self._jaccard_similarity(ideal_answer, student_answer)
        overlap = self._keyword_overlap(ideal_answer, student_answer)
        
        return {
            'semantic_score': normalized_score,
            'cosine_similarity': cos_sim,
            'jaccard_similarity': jaccard,
            'keyword_overlap': overlap,
            'confidence': confidence,
            'method': 'sbert' if self.model is not None else 'tfidf_fallback'
        }
    
    def _calculate_confidence(self, emb1: torch.Tensor, emb2: torch.Tensor) -> float:
        """
        Calculate confidence in similarity score
        Based on vector norms and angle consistency
        """
        # Check vector norms (zero vectors indicate poor text)
        norm1 = torch.norm(emb1).item()
        norm2 = torch.norm(emb2).item()
        
        if norm1 < 0.01 or norm2 < 0.01:
            return 0.3  # Low confidence for empty/near-empty vectors
        
        # Check if vectors are very sparse (TF-IDF case)
        sparsity1 = (emb1 == 0).float().mean().item()
        sparsity2 = (emb2 == 0).float().mean().item()
        
        if sparsity1 > 0.9 or sparsity2 > 0.9:
            return 0.5  # Medium confidence for sparse vectors
        
        # High confidence for dense, non-zero vectors
        return 0.9
    
    def _jaccard_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate Jaccard similarity between word sets
        Formula: J(A,B) = |A ∩ B| / |A ∪ B|
        """
        # Tokenize and remove stopwords
        words1 = set(self._tokenize(text1))
        words2 = set(self._tokenize(text2))
        
        # Remove common stopwords
        stopwords = {'the', 'and', 'is', 'in', 'to', 'of', 'for', 'with', 'on', 'that'}
        words1 = words1 - stopwords
        words2 = words2 - stopwords
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def _keyword_overlap(self, text1: str, text2: str, top_n=10) -> float:
        """
        Calculate overlap of most important keywords
        """
        # Extract keywords (simplified - in practice use RAKE or KeyBERT)
        keywords1 = self._extract_keywords(text1, top_n)
        keywords2 = self._extract_keywords(text2, top_n)
        
        if not keywords1 or not keywords2:
            return 0.0
        
        overlap = len(set(keywords1).intersection(set(keywords2)))
        return overlap / len(keywords1)
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization"""
        text = text.lower()
        # Remove punctuation
        text = ''.join(c if c.isalnum() or c.isspace() else ' ' for c in text)
        return [word for word in text.split() if len(word) > 2]
    
    def _extract_keywords(self, text: str, top_n: int = 10) -> List[str]:
        """Extract keywords using simple frequency"""
        words = self._tokenize(text)
        
        # Remove common words
        common_words = {'the', 'and', 'is', 'in', 'to', 'of', 'for', 'with', 'on', 'that',
                       'this', 'are', 'as', 'be', 'by', 'or', 'an', 'it', 'from', 'was'}
        words = [w for w in words if w not in common_words]
        
        # Count frequency
        from collections import Counter
        word_counts = Counter(words)
        
        # Get top N
        keywords = [word for word, count in word_counts.most_common(top_n)]
        
        return keywords
    
    def fine_tune(self, pairs: List[Tuple[str, str]], labels: List[float], 
                 epochs: int = 3, batch_size: int = 16):
        """
        Fine-tune SBERT on custom data
        Loss: CosineSimilarityLoss
        """
        if self.model is None:
            print("⚠ Cannot fine-tune: SBERT model not loaded")
            return
        
        from sentence_transformers import InputExample, losses, datasets
        from torch.utils.data import DataLoader
        
        # Create training examples
        train_examples = []
        for (text1, text2), label in zip(pairs, labels):
            train_examples.append(InputExample(
                texts=[text1, text2],
                label=float(label)
            ))
        
        # Create dataloader
        train_dataloader = DataLoader(
            train_examples,
            shuffle=True,
            batch_size=batch_size
        )
        
        # Define loss
        train_loss = losses.CosineSimilarityLoss(model=self.model)
        
        # Fine-tune
        self.model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=epochs,
            warmup_steps=100,
            show_progress_bar=True
        )
        
        print("✓ SBERT fine-tuning completed")