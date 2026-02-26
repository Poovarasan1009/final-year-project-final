"""
neural_evaluator.py
───────────────────
Custom Deep Learning model for answer evaluation.

Architecture:  Multi-Layer Perceptron (MLP) built with PyTorch
Input:         Feature vector from the existing 4-layer evaluator
Output:        Single regression score (0-100)

This file defines:
1. EvaluationNN       — the PyTorch neural network
2. NeuralEvaluator    — wrapper class for prediction in production
"""

import os
import torch
import torch.nn as nn
import numpy as np

# ─────────────────────────────────────────────────────────────────────
# 1. Neural Network Architecture
# ─────────────────────────────────────────────────────────────────────
class EvaluationNN(nn.Module):
    """
    Deep Learning model for descriptive answer evaluation.
    
    Architecture:
        Input  (7 features) → Dense(64) → ReLU → Dropout(0.3)
                             → Dense(32) → ReLU → Dropout(0.2)
                             → Dense(16) → ReLU
                             → Dense(1)  → Sigmoid × 100
    
    Features:
        0: concept_score       (0-1)  — keyword/synonym coverage
        1: semantic_score      (0-1)  — SBERT cosine similarity
        2: structure_score     (0-1)  — grammar & coherence
        3: completeness_score  (0-1)  — coverage of ideal answer
        4: word_count_ratio    (0-2)  — len(student) / len(ideal)
        5: accuracy_boost      (0-1)  — synonym + n-gram engine
        6: concept_phrase_score(0-1)  — domain phrase matching
    """

    def __init__(self, input_dim=7, hidden_dims=None, dropout_rates=None):
        super(EvaluationNN, self).__init__()

        if hidden_dims is None:
            hidden_dims = [64, 32, 16]
        if dropout_rates is None:
            dropout_rates = [0.3, 0.2, 0.0]

        layers = []
        prev_dim = input_dim

        for i, (h_dim, drop) in enumerate(zip(hidden_dims, dropout_rates)):
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            if drop > 0:
                layers.append(nn.Dropout(drop))
            prev_dim = h_dim

        # Output layer: single neuron for regression
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())  # constrain output to 0-1

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass: returns score in range [0, 1]."""
        return self.network(x)


# ─────────────────────────────────────────────────────────────────────
# 2. Production Wrapper
# ─────────────────────────────────────────────────────────────────────
class NeuralEvaluator:
    """
    Production wrapper for the trained EvaluationNN model.
    Loads saved weights and provides a simple predict() interface.
    """

    def __init__(self, model_path=None):
        self.device = torch.device("cpu")  # CPU-only for compatibility
        self.model = EvaluationNN()
        self.model_loaded = False

        if model_path is None:
            # Default path: same directory as this file
            model_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "trained_models",
                "evaluation_nn.pth"
            )

        if os.path.exists(model_path):
            try:
                checkpoint = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(checkpoint["model_state_dict"])
                self.model.eval()
                self.model_loaded = True
                self.metadata = checkpoint.get("metadata", {})
                print(f"[DL] Neural evaluator loaded from {model_path}")
                print(f"     Training accuracy: {self.metadata.get('best_val_mae', 'N/A')}")
            except Exception as e:
                print(f"[WARN] Failed to load neural model: {e}")
                self.model_loaded = False
        else:
            print(f"[INFO] No trained model found at {model_path}")
            print(f"       Run train_model.py first to train the neural evaluator.")

    def predict(self, features: dict) -> float:
        """
        Predict a score from 0-100 given feature dictionary.
        
        Args:
            features: dict with keys:
                - concept_score (float 0-1)
                - semantic_score (float 0-1)
                - structure_score (float 0-1)
                - completeness_score (float 0-1)
                - word_count_ratio (float 0-2+)
                - accuracy_boost (float 0-1)
                - concept_phrase_score (float 0-1)

        Returns:
            float: predicted score 0-100
        """
        if not self.model_loaded:
            return None

        # Build feature vector in the correct order
        feature_vector = np.array([
            features.get("concept_score", 0.0),
            features.get("semantic_score", 0.0),
            features.get("structure_score", 0.0),
            features.get("completeness_score", 0.0),
            min(features.get("word_count_ratio", 0.0), 2.0),  # clip
            features.get("accuracy_boost", 0.0),
            features.get("concept_phrase_score", 0.0),
        ], dtype=np.float32)

        tensor = torch.tensor(feature_vector, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            output = self.model(tensor)
            score = output.item() * 100.0  # scale sigmoid to 0-100

        return round(score, 2)

    def is_loaded(self) -> bool:
        """Check if the model weights are loaded and ready."""
        return self.model_loaded
