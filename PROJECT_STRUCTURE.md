# Project Structure Documentation

This document provides a detailed overview of the folder structure and file organization for the **Answer Evaluation System**.

## 📂 Root Directory
*   `start_server.py`: The main script to launch the application cleanly. It installs requirements, activates the environment, starts the server, and opens the browser.
*   `main.py`: Interactive CLI tool to run specific modules, start the server, or run the mentor demo.
*   `ALGORITHM_README.md`: Detailed mathematical and conceptual documentation of the 4-layer evaluation engine.
*   `PROJECT_STRUCTURE.md`: This file. High-level architecture and data flow.
*   `requirements.txt`: Python dependencies (FastAPI, PyTorch, SentenceTransformers, etc.)

## 🧠 Advanced_Core/ (The AI Engine & Training Pipeline)
*This is the heart of the project where the Machine Learning models live and learn.*

### 1. The Evaluation Algorithms (`advanced_evaluator.py`, `neural_evaluator.py`)
*   **Algorithms Used**: 
    1.  **SBERT (all-MiniLM-L6-v2) & TF-IDF**: Used for semantic cosine similarity.
    2.  **PyTorch Feed-Forward Neural Network (MLP)**: A custom-trained Deep Learning model with dropout and ReLU activations that predicts the final score based on extracted features.
    3.  **Heuristic NLP (NLTK)**: Used for structural grammar analysis and concept matching via Jaccard index.
*   **How it Works**: It extracts 7 specific feature numbers from a student's answer (Concept coverage, Semantic similarity, Sentence count, Word count, Connector density, Completeness type-score, and Sequence score) and feeds them into the Neural Network to get a final grade out of 100.

### 2. The Training Process (`train_model.py`, `generate_training_data.py`)
*   **Where do we get the data?** We use a synthetic data generator (`generate_training_data.py`) to create thousands of simulated Q&A pairs (ranging from perfect answers to terrible ones) and save them to `Advanced_Core/Saved_Models/training_dataset.csv`.
*   **How do we train the model?** The `train_model.py` script loads the CSV, converts the text into the 7-feature arrays, and pushes them through a PyTorch Neural Network. The network learns the difference between a high-scoring answer and a low-scoring one using Mean Squared Error (MSE) loss and the Adam optimizer.
*   **Where are the Models saved?** The trained neural network weights are saved precisely at `Advanced_Core/Saved_Models/evaluation_model.pth`.

### 3. Model Metrics & Graphs
*   When `train_model.py` finishes, it automatically generates a performance graph showing the **Training Loss vs Validation Loss**.
*   These graphs (metrics) are saved as `Advanced_Core/Saved_Models/training_curves.png`. This proves the model actually learned and didn't just overfit.

## 🎨 Frontend/ (User Interface)
*   **Student UI (`student_question.html`)**: Features real-time character counting, Speech-to-Text dictation, and an interactive **Chart.js Radar Graph** showing the 4-layer AI breakdown of their answer.
*   **Teacher UI (`teacher_analytics.html`)**: Features aggregate Data Visualization (Bar charts for topic performance, Doughnut charts for AI confidence) populated dynamically from the SQLite database.

## 🚀 Production_Deployment/ (Backend Server)
*   `fastapi_app.py`: The REST API. It handles JWT authentication, routes data between SQLite and the `Advanced_Core` AI, and serves the HTML templates.

## 🛠️ Utilities/ (Database & Storage)
*   `database_manager.py`: Handles everything SQLite.
*   **Data Stored**: `Data/evaluations.db`. Stores hashed passwords, users (Admin, Teacher, Student), created questions, and every single student submission with its detailed 4-layer metric breakdown.

---

## 🔄 End-to-End Data Flow

1.  **Data Generation & Training (Offline)**: Run `train_model.py` -> Generates CSV Data -> Extracts Features -> Trains PyTorch Model -> Saves `.pth` weights and `.png` graphs.
2.  **Web Request (Online)**: Student submits answer via `student_question.html`.
3.  **API Routing**: `fastapi_app.py` receives the string.
4.  **AI Inference**: 
    *   `advanced_evaluator.py` runs SBERT / TF-IDF / NLTK to get the 7 features.
    *   `neural_evaluator.py` loads the `.pth` model and predicts the score.
5.  **Storage**: `database_manager.py` saves the answer and the score to SQLite.
6.  **Analytics**: The Teacher logs in and the system queries SQLite to generate `Chart.js` graphs comparing class performance.
