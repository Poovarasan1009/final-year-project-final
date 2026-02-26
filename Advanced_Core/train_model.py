"""
train_model.py
──────────────
Complete training pipeline for the custom Deep Learning answer evaluator.

Pipeline:
1. Load training_data.csv (1000 samples)
2. Extract features using the existing 4-layer evaluator (SBERT, etc.)
3. Split into train/validation sets (80/20)
4. Train a PyTorch MLP for 100 epochs
5. Save the best model checkpoint
6. Generate training loss curves for documentation
"""

import os
import sys
import csv
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

# ── Ensure project root is on PYTHONPATH ──
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from Advanced_Core.neural_evaluator import EvaluationNN

# ─────────────────────────────────────────────────────────────────────
# 1. Feature Extraction
# ─────────────────────────────────────────────────────────────────────

def extract_features_batch(csv_path, cache_path=None):
    """
    Extract features from each row in the CSV using the 4-layer evaluator.
    Returns a list of (feature_vector, human_score) tuples.
    
    Features extracted per sample:
        0: concept_score       — keyword/synonym coverage
        1: semantic_score      — SBERT cosine similarity
        2: structure_score     — grammar & coherence
        3: completeness_score  — coverage of ideal answer
        4: word_count_ratio    — len(student) / len(ideal)
        5: accuracy_boost      — synonym + n-gram engine
        6: concept_phrase_score— domain phrase matching
    """
    # Check for cached features first
    if cache_path and os.path.exists(cache_path):
        print(f"[INFO] Loading cached features from {cache_path}")
        data = np.load(cache_path, allow_pickle=True)
        return data["features"], data["scores"]

    print("[INFO] Initializing evaluator for feature extraction...")
    print("       (This loads the SBERT model — may take a moment)")

    # Import evaluator here to avoid circular imports
    from Advanced_Core.advanced_evaluator import AdvancedAnswerEvaluator
    evaluator = AdvancedAnswerEvaluator()

    # Read CSV
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    print(f"[INFO] Extracting features from {len(rows)} samples...")

    features_list = []
    scores_list = []
    start_time = time.time()

    for i, row in enumerate(rows):
        question = row["question"]
        ideal = row["ideal_answer"]
        student = row["student_answer"]
        human_score = float(row["human_score"])

        try:
            # Run the full evaluator to get all layer scores
            result = evaluator.evaluate(question, ideal, student)

            layer = result.get("layer_scores", {})
            acc = result.get("accuracy_details", {})

            # Build feature vector
            concept_score = layer.get("conceptual", 0.0)
            semantic_score = layer.get("semantic", 0.0)
            structure_score = layer.get("structural", 0.0)
            completeness_score = layer.get("completeness", 0.0)

            # Word count ratio
            ideal_words = len(ideal.split())
            student_words = len(student.split())
            word_count_ratio = min(student_words / max(ideal_words, 1), 2.0)

            accuracy_boost = acc.get("accuracy_score", 0.0)
            concept_phrase_score = acc.get("concept_phrase_score", 0.0)

            feature_vec = [
                concept_score,
                semantic_score,
                structure_score,
                completeness_score,
                word_count_ratio,
                accuracy_boost,
                concept_phrase_score,
            ]

            features_list.append(feature_vec)
            scores_list.append(human_score / 100.0)  # normalize to 0-1

        except Exception as e:
            print(f"   [WARN] Row {i} failed: {e}")
            continue

        if (i + 1) % 50 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            remaining = (len(rows) - i - 1) / rate
            print(f"   [{i+1}/{len(rows)}] {rate:.1f} samples/sec, ~{remaining:.0f}s remaining")

    features = np.array(features_list, dtype=np.float32)
    scores = np.array(scores_list, dtype=np.float32)

    # Cache for future runs
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.savez(cache_path, features=features, scores=scores)
        print(f"[INFO] Features cached to {cache_path}")

    elapsed = time.time() - start_time
    print(f"[DONE] Extracted {len(features)} feature vectors in {elapsed:.1f}s")

    return features, scores


# ─────────────────────────────────────────────────────────────────────
# 2. PyTorch Dataset
# ─────────────────────────────────────────────────────────────────────

class EvaluationDataset(Dataset):
    """Custom PyTorch Dataset for loading feature/score pairs."""

    def __init__(self, features, scores):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.scores = torch.tensor(scores, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.scores[idx]


# ─────────────────────────────────────────────────────────────────────
# 3. Training Loop
# ─────────────────────────────────────────────────────────────────────

def train_model(features, scores, config=None):
    """
    Train the EvaluationNN model.
    
    Args:
        features: np.array of shape (N, 7)
        scores:   np.array of shape (N,) in range [0, 1]
        config:   dict of hyperparameters
    
    Returns:
        tuple: (trained_model, training_history)
    """
    if config is None:
        config = {
            "epochs": 100,
            "batch_size": 32,
            "learning_rate": 0.001,
            "weight_decay": 1e-4,
            "val_split": 0.2,
            "patience": 15,  # early stopping
        }

    print("\n" + "=" * 60)
    print("  TRAINING CONFIGURATION")
    print("=" * 60)
    for k, v in config.items():
        print(f"  {k:20s} : {v}")
    print("=" * 60)

    # Create dataset
    dataset = EvaluationDataset(features, scores)

    # Split into train/val
    val_size = int(len(dataset) * config["val_split"])
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)

    print(f"\n  Training samples  : {train_size}")
    print(f"  Validation samples: {val_size}")

    # Initialize model
    model = EvaluationNN(input_dim=7)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"]
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    # Training history
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_mae": [],
        "val_mae": [],
    }

    best_val_loss = float("inf")
    best_model_state = None
    patience_counter = 0

    print(f"\n{'Epoch':>6s} | {'Train Loss':>12s} | {'Val Loss':>12s} | {'Train MAE':>10s} | {'Val MAE':>10s} | {'LR':>10s}")
    print("-" * 75)

    for epoch in range(config["epochs"]):
        # ── Training Phase ──
        model.train()
        train_losses = []
        train_maes = []

        for batch_features, batch_scores in train_loader:
            optimizer.zero_grad()
            predictions = model(batch_features)
            loss = criterion(predictions, batch_scores)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            mae = torch.abs(predictions - batch_scores).mean().item() * 100
            train_maes.append(mae)

        # ── Validation Phase ──
        model.eval()
        val_losses = []
        val_maes = []

        with torch.no_grad():
            for batch_features, batch_scores in val_loader:
                predictions = model(batch_features)
                loss = criterion(predictions, batch_scores)
                val_losses.append(loss.item())
                mae = torch.abs(predictions - batch_scores).mean().item() * 100
                val_maes.append(mae)

        # Aggregate metrics
        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)
        avg_train_mae = np.mean(train_maes)
        avg_val_mae = np.mean(val_maes)
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["train_mae"].append(avg_train_mae)
        history["val_mae"].append(avg_val_mae)

        # Print progress every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"{epoch+1:6d} | {avg_train_loss:12.6f} | {avg_val_loss:12.6f} | {avg_train_mae:10.2f} | {avg_val_mae:10.2f} | {current_lr:10.6f}")

        # Learning rate scheduling
        scheduler.step(avg_val_loss)

        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config["patience"]:
                print(f"\n[EARLY STOP] No improvement for {config['patience']} epochs. Stopping.")
                break

    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)

    print(f"\n{'=' * 60}")
    print(f"  TRAINING COMPLETE")
    print(f"  Best Validation Loss : {best_val_loss:.6f}")
    print(f"  Best Validation MAE  : {min(history['val_mae']):.2f} points")
    print(f"  Epochs completed     : {len(history['train_loss'])}")
    print(f"{'=' * 60}")

    return model, history, config


# ─────────────────────────────────────────────────────────────────────
# 4. Save & Plot
# ─────────────────────────────────────────────────────────────────────

def save_model(model, history, config, save_dir):
    """Save model checkpoint and training metadata."""
    os.makedirs(save_dir, exist_ok=True)

    # Save model weights
    model_path = os.path.join(save_dir, "evaluation_nn.pth")
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "metadata": {
            "input_dim": 7,
            "hidden_dims": [64, 32, 16],
            "best_val_mae": min(history["val_mae"]),
            "best_val_loss": min(history["val_loss"]),
            "epochs_trained": len(history["train_loss"]),
            "config": config,
            "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
    }
    torch.save(checkpoint, model_path)
    print(f"[SAVED] Model weights -> {model_path}")

    # Save history for plotting
    history_path = os.path.join(save_dir, "training_history.json")
    serializable_history = {k: [float(v) for v in vals] for k, vals in history.items()}
    with open(history_path, "w") as f:
        json.dump(serializable_history, f, indent=2)
    print(f"[SAVED] Training history -> {history_path}")

    return model_path


def plot_training_curves(history, save_dir):
    """Generate training/validation loss curves for documentation."""
    try:
        import matplotlib
        matplotlib.use("Agg")  # non-interactive backend
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Loss Curve
        axes[0].plot(history["train_loss"], label="Training Loss", color="#4A90D9", linewidth=2)
        axes[0].plot(history["val_loss"], label="Validation Loss", color="#E74C3C", linewidth=2)
        axes[0].set_xlabel("Epoch", fontsize=12)
        axes[0].set_ylabel("MSE Loss", fontsize=12)
        axes[0].set_title("Training vs Validation Loss", fontsize=14, fontweight="bold")
        axes[0].legend(fontsize=11)
        axes[0].grid(True, alpha=0.3)

        # MAE Curve
        axes[1].plot(history["train_mae"], label="Training MAE", color="#27AE60", linewidth=2)
        axes[1].plot(history["val_mae"], label="Validation MAE", color="#F39C12", linewidth=2)
        axes[1].set_xlabel("Epoch", fontsize=12)
        axes[1].set_ylabel("Mean Absolute Error (points)", fontsize=12)
        axes[1].set_title("Training vs Validation MAE", fontsize=14, fontweight="bold")
        axes[1].legend(fontsize=11)
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = os.path.join(save_dir, "training_curves.png")
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[SAVED] Training curves -> {plot_path}")
        return plot_path

    except ImportError:
        print("[WARN] matplotlib not available. Skipping plot generation.")
        return None


# ─────────────────────────────────────────────────────────────────────
# 5. Main Entry Point
# ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  DEEP LEARNING MODEL TRAINING PIPELINE")
    print("  Answer Evaluation Neural Network")
    print("=" * 60)

    # Paths
    csv_path = os.path.join(ROOT, "Real_Dataset", "training_data.csv")
    cache_path = os.path.join(ROOT, "Advanced_Core", "trained_models", "feature_cache.npz")
    save_dir = os.path.join(ROOT, "Advanced_Core", "trained_models")

    if not os.path.exists(csv_path):
        print(f"[ERROR] Training data not found at {csv_path}")
        print(f"        Run generate_training_data.py first!")
        sys.exit(1)

    # Step 1: Extract features
    print("\n--- STEP 1: Feature Extraction ---")
    features, scores = extract_features_batch(csv_path, cache_path)
    print(f"  Feature shape: {features.shape}")
    print(f"  Score range  : [{scores.min():.2f}, {scores.max():.2f}]")

    # Step 2: Train model
    print("\n--- STEP 2: Model Training ---")
    model, history, config = train_model(features, scores)

    # Step 3: Save model
    print("\n--- STEP 3: Saving Model ---")
    model_path = save_model(model, history, config, save_dir)

    # Step 4: Generate plots
    print("\n--- STEP 4: Generating Plots ---")
    plot_path = plot_training_curves(history, save_dir)

    print("\n" + "=" * 60)
    print("  ALL DONE!")
    print(f"  Model saved to: {model_path}")
    if plot_path:
        print(f"  Plots saved to: {plot_path}")
    print("  You can now use this model in the evaluator.")
    print("=" * 60)


if __name__ == "__main__":
    main()
