# Project Structure Documentation

This document provides a detailed overview of the folder structure and file organization for the **Answer Evaluation System**.

## 📂 Root Directory
*   `main.py`: The entry point for running the application.
*   `ALGORITHM_README.md`: Detailed documentation of the AI algorithms, formulas, and novelty.
*   `requirements.txt`: List of Python dependencies.

## 🧠 Advanced_Core/
*Contains the AI logic and evaluation engine.*
*   `advanced_evaluator.py`: **The Brain.** Contains the 4-layer evaluation class (`AdvancedAnswerEvaluator`), SBERT model loading, and scoring logic.
*   `__init__.py`: Makes the folder a Python package.

## 🎨 Frontend/
*Contains all user interface code.*
*   `static/`: CSS, JavaScript, and images.
*   **`templates/`**: HTML files for the web interface.
    *   **Student Pages**:
        *   `student_question.html`: The main interface where students answer questions. (Features: Voice Input, Radar Chart).
        *   `student_dashboard.html`: Student's homepage.
    *   **Teacher Pages**:
        *   `teacher_dashboard.html`: Teacher's command center with analytics.
        *   `teacher_analytics.html`: Detailed metrics and class performance.
    *   **Common**:
        *   `login.html`, `index.html`.

## 🚀 Production_Deployment/
*Contains the backend server code.*
*   `fastapi_app.py`: The **FastAPI Server**. Handles all API requests, routing, and connects the Frontend to the AI Core.
*   `auth_system.py`: Manages user login, registration, and JWT token security.

## 🛠️ Utilities/
*Helper scripts and database management.*
*   `database_manager.py`: Handles all SQLite database operations (saving answers, fetching analytics, user management).

## 📊 Real_Dataset/
*Data storage.*
*   `sample_dataset.csv`: Example Q&A pairs for testing.
*   `school.db`: The SQLite database file storing users, questions, and answers.

## 🧪 Tests/
*Verification and quality assurance.*
*   `verify_enhancements.py`: Script to check if new features are working.
*   `verify_submission.py`: Tests the answer submission pipeline.

---

## Key Data Flow
1.  **User** interacts with `Frontend/templates/*.html`.
2.  Browser sends API request to `Production_Deployment/fastapi_app.py`.
3.  Server calls `Advanced_Core/advanced_evaluator.py` to grade the answer.
4.  Result is saved via `Utilities/database_manager.py` to `school.db`.
5.  Results are sent back to the Frontend for display.
