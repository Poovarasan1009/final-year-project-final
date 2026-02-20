import numpy as np
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

class LDAModel:
    """
    Implements Latent Dirichlet Allocation using Collapsed Gibbs Sampling
    Mathematical Formula: P(z_i=k|z_{-i}, w) ∝ (n_{k,-i}^{(d)} + α) * (n_{k,-i}^{(w)} + β) / (n_{k,-i}^{(·)} + Vβ)
    """
    
    def __init__(self, num_topics=3, alpha=50/3, beta=0.01, iterations=1000):
        self.K = num_topics           # Number of topics
        self.alpha = alpha            # Dirichlet prior for document-topic
        self.beta = beta              # Dirichlet prior for topic-word
        self.iterations = iterations  # Gibbs sampling iterations
        
    def fit(self, documents):
        """
        Train LDA model using Collapsed Gibbs Sampling
        """
        # Preprocess: tokenize and create vocabulary
        self.vocab = self._create_vocabulary(documents)
        self.V = len(self.vocab)  # Vocabulary size
        self.word_to_id = {word: idx for idx, word in enumerate(self.vocab)}
        
        # Convert documents to word IDs
        doc_word_ids = []
        for doc in documents:
            tokens = self._tokenize(doc)
            word_ids = [self.word_to_id[word] for word in tokens if word in self.word_to_id]
            doc_word_ids.append(word_ids)
        
        self.D = len(doc_word_ids)  # Number of documents
        
        # Initialize counts
        # n_dk: document-topic count
        self.n_dk = np.zeros((self.D, self.K), dtype=int)
        # n_kw: topic-word count
        self.n_kw = np.zeros((self.K, self.V), dtype=int)
        # n_k: total words per topic
        self.n_k = np.zeros(self.K, dtype=int)
        
        # Randomly assign topics
        self.z = []  # Topic assignments for each word
        for d, word_ids in enumerate(doc_word_ids):
            doc_topics = []
            for w in word_ids:
                # Random topic assignment
                k = np.random.randint(0, self.K)
                doc_topics.append(k)
                
                # Update counts
                self.n_dk[d, k] += 1
                self.n_kw[k, w] += 1
                self.n_k[k] += 1
            self.z.append(doc_topics)
        
        # Gibbs sampling
        for iteration in range(self.iterations):
            for d in range(self.D):
                for i, w in enumerate(doc_word_ids[d]):
                    k_old = self.z[d][i]
                    
                    # Decrement counts
                    self.n_dk[d, k_old] -= 1
                    self.n_kw[k_old, w] -= 1
                    self.n_k[k_old] -= 1
                    
                    # Calculate probability for each topic
                    probs = np.zeros(self.K)
                    for k in range(self.K):
                        # Gibbs sampling formula
                        prob = (self.n_dk[d, k] + self.alpha) * \
                               (self.n_kw[k, w] + self.beta) / \
                               (self.n_k[k] + self.V * self.beta)
                        probs[k] = prob
                    
                    # Normalize probabilities
                    probs_sum = np.sum(probs)
                    if probs_sum > 0:
                        probs = probs / probs_sum
                    else:
                        probs = np.ones(self.K) / self.K
                    
                    # Sample new topic
                    k_new = np.random.choice(self.K, p=probs)
                    self.z[d][i] = k_new
                    
                    # Increment counts
                    self.n_dk[d, k_new] += 1
                    self.n_kw[k_new, w] += 1
                    self.n_k[k_new] += 1
            
            # Optional: Print progress
            if (iteration + 1) % 100 == 0:
                perplexity = self._calculate_perplexity(doc_word_ids)
                print(f"Iteration {iteration+1}/{self.iterations}, Perplexity: {perplexity:.2f}")
        
        return self
    
    def get_document_topic_distribution(self, document):
        """
        Get topic distribution for a new document
        Formula: θ_dk = (n_dk + α) / (n_d + Kα)
        """
        tokens = self._tokenize(document)
        word_ids = [self.word_to_id[word] for word in tokens if word in self.word_to_id]
        
        if not word_ids:
            return np.ones(self.K) / self.K
        
        # Initialize counts for this document
        doc_topic_counts = np.zeros(self.K)
        
        # Assign words to topics using trained model
        for w in word_ids:
            if w >= self.V:  # Word not in vocabulary
                continue
                
            # Calculate probability for each topic
            probs = np.zeros(self.K)
            for k in range(self.K):
                prob = (self.n_kw[k, w] + self.beta) / (self.n_k[k] + self.V * self.beta)
                probs[k] = prob
            
            # Assign to most probable topic
            k_best = np.argmax(probs)
            doc_topic_counts[k_best] += 1
        
        # Dirichlet posterior
        theta = (doc_topic_counts + self.alpha) / (len(word_ids) + self.K * self.alpha)
        
        return theta
    
    def get_topic_words(self, top_n=10):
        """
        Get most probable words for each topic
        """
        topic_words = []
        for k in range(self.K):
            # Get word probabilities for this topic
            word_probs = (self.n_kw[k] + self.beta) / (self.n_k[k] + self.V * self.beta)
            
            # Get top N words
            top_indices = np.argsort(word_probs)[-top_n:][::-1]
            words = [(self.vocab[idx], word_probs[idx]) for idx in top_indices]
            topic_words.append(words)
        
        return topic_words
    
    def _create_vocabulary(self, documents, min_count=2):
        """Create vocabulary from documents"""
        word_counts = defaultdict(int)
        for doc in documents:
            tokens = self._tokenize(doc)
            for token in tokens:
                word_counts[token] += 1
        
        # Filter by minimum count
        vocab = [word for word, count in word_counts.items() 
                if count >= min_count and len(word) > 2]
        
        return vocab
    
    def _tokenize(self, text):
        """Simple tokenization"""
        text = text.lower()
        # Remove punctuation and split
        tokens = []
        current_token = []
        for char in text:
            if char.isalnum():
                current_token.append(char)
            elif current_token:
                tokens.append(''.join(current_token))
                current_token = []
        if current_token:
            tokens.append(''.join(current_token))
        
        return tokens
    
    def _calculate_perplexity(self, doc_word_ids):
        """
        Calculate perplexity: exp(-(1/N) * Σ log p(w|d))
        Lower perplexity = better model
        """
        total_log_prob = 0
        total_words = 0
        
        for d, word_ids in enumerate(doc_word_ids):
            for w in word_ids:
                # Calculate p(w|d) = Σ_k p(w|k) * p(k|d)
                prob_w_given_d = 0
                for k in range(self.K):
                    # p(w|k)
                    p_w_k = (self.n_kw[k, w] + self.beta) / (self.n_k[k] + self.V * self.beta)
                    # p(k|d)
                    p_k_d = (self.n_dk[d, k] + self.alpha) / (len(word_ids) + self.K * self.alpha)
                    prob_w_given_d += p_w_k * p_k_d
                
                if prob_w_given_d > 0:
                    total_log_prob += np.log(prob_w_given_d)
                total_words += 1
        
        if total_words == 0:
            return float('inf')
        
        perplexity = np.exp(-total_log_prob / total_words)
        return perplexity