import sqlite3
import pandas as pd
import os
from datetime import datetime
import json
import hashlib
import secrets
from contextlib import contextmanager


def get_grade_band(score: float) -> dict:
    """Return grade band dict for a given percentage score."""
    if score >= 90:
        return {'grade': 'A+', 'label': 'Outstanding'}
    elif score >= 80:
        return {'grade': 'A',  'label': 'Excellent'}
    elif score >= 70:
        return {'grade': 'B',  'label': 'Good'}
    elif score >= 60:
        return {'grade': 'C',  'label': 'Average'}
    elif score >= 50:
        return {'grade': 'D',  'label': 'Below Average'}
    elif score >= 35:
        return {'grade': 'E',  'label': 'Poor'}
    else:
        return {'grade': 'F',  'label': 'Fail'}


class DatabaseManager:
    def __init__(self, db_path="Data/evaluations.db"):
        self.db_path = db_path
        self.conn = None
        self.cursor = None

    def validate_answer(self, text: str) -> tuple:
        """
        Validate student answer before sending to evaluator.
        Returns (is_valid: bool, message: str)
        """
        if not text or not text.strip():
            return False, 'Answer cannot be empty.'
        if len(text.strip()) < 5:
            return False, 'Answer is too short. Please write at least a sentence.'
        if len(text.strip()) > 5000:
            return False, 'Answer is too long. Please limit to 5000 characters.'
        # Check if answer is just repeated characters / garbage
        unique_chars = len(set(text.lower().replace(' ', '')))
        if unique_chars < 4:
            return False, 'Answer does not contain meaningful content.'
        return True, 'OK'

    def _save_to_dataset(self, filename: str, data: dict):
        """Append data to a CSV file in Real_Dataset folder"""
        try:
            # Ensure folder exists
            os.makedirs("Real_Dataset", exist_ok=True)
            file_path = f"Real_Dataset/{filename}"
            
            df = pd.DataFrame([data])
            
            # Append if exists, else create with header
            if os.path.exists(file_path):
                df.to_csv(file_path, mode='a', header=False, index=False)
            else:
                df.to_csv(file_path, mode='w', header=True, index=False)
            print(f"✓ Saved to {filename}")
        except Exception as e:
            print(f"⚠ Failed to save to dataset: {e}")
    
    def _get_conn(self):
        """Get a thread-safe SQLite connection."""
        return sqlite3.connect(
            self.db_path,
            check_same_thread=False,   # allow access from multiple threads
            timeout=30                  # wait up to 30s if DB is locked
        )

    def initialize(self):
        """Initialize database with user roles and questions"""
        self.conn = self._get_conn()
        self.cursor = self.conn.cursor()
        
        # Users table (Admin, Teacher, Student)
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL,  -- 'admin', 'teacher', 'student'
                full_name TEXT,
                email TEXT,
                created_at TEXT,
                last_login TEXT
            )
        ''')
        
        # Teachers table (extended info)
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS teachers (
                user_id INTEGER PRIMARY KEY,
                department TEXT,
                subject TEXT,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        ''')
        
        # Students table
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS students (
                user_id INTEGER PRIMARY KEY,
                roll_number TEXT UNIQUE,
                class TEXT,
                year INTEGER,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        ''')
        
        # Questions table (created by teachers)
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS questions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question_text TEXT NOT NULL,
                subject TEXT,
                topic TEXT,
                difficulty TEXT,  -- easy, medium, hard
                marks INTEGER DEFAULT 10,
                created_by INTEGER,  -- teacher user_id
                created_at TEXT,
                ideal_answer TEXT,
                keywords TEXT,  -- JSON list of keywords
                FOREIGN KEY (created_by) REFERENCES users(id)
            )
        ''')
        
        # Student answers table
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS student_answers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id INTEGER,
                question_id INTEGER,
                answer_text TEXT NOT NULL,
                submitted_at TEXT,
                evaluated_at TEXT,
                final_score REAL,
                confidence REAL,
                feedback TEXT,
                layer_scores TEXT,  -- JSON object
                is_evaluated INTEGER DEFAULT 0,
                FOREIGN KEY (student_id) REFERENCES users(id),
                FOREIGN KEY (question_id) REFERENCES questions(id)
            )
        ''')
        
        # Classes/Assignments table
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS assignments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                description TEXT,
                teacher_id INTEGER,
                created_at TEXT,
                due_date TEXT,
                is_active INTEGER DEFAULT 1,
                FOREIGN KEY (teacher_id) REFERENCES users(id)
            )
        ''')
        
        # Assignment questions (many-to-many)
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS assignment_questions (
                assignment_id INTEGER,
                question_id INTEGER,
                sequence INTEGER,
                PRIMARY KEY (assignment_id, question_id),
                FOREIGN KEY (assignment_id) REFERENCES assignments(id),
                FOREIGN KEY (question_id) REFERENCES questions(id)
            )
        ''')
        
        # Student assignments (track completion)
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS student_assignments (
                student_id INTEGER,
                assignment_id INTEGER,
                started_at TEXT,
                completed_at TEXT,
                total_score REAL,
                status TEXT DEFAULT 'not_started',
                PRIMARY KEY (student_id, assignment_id),
                FOREIGN KEY (student_id) REFERENCES users(id),
                FOREIGN KEY (assignment_id) REFERENCES assignments(id)
            )
        ''')

        # Exams table
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS exams (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                description TEXT,
                teacher_id INTEGER,
                start_time TEXT,
                end_time TEXT,
                duration_minutes INTEGER,
                is_active INTEGER DEFAULT 1,
                strict_mode INTEGER DEFAULT 1,
                FOREIGN KEY (teacher_id) REFERENCES users(id)
            )
        ''')

        # Exam questions (many-to-many)
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS exam_questions (
                exam_id INTEGER,
                question_id INTEGER,
                sequence INTEGER,
                PRIMARY KEY (exam_id, question_id),
                FOREIGN KEY (exam_id) REFERENCES exams(id),
                FOREIGN KEY (question_id) REFERENCES questions(id)
            )
        ''')

        # Student exams (track completion and proctoring)
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS student_exams (
                student_id INTEGER,
                exam_id INTEGER,
                started_at TEXT,
                submitted_at TEXT,
                total_score REAL,
                status TEXT DEFAULT 'not_started',
                proctoring_score REAL DEFAULT 100,
                violation_count INTEGER DEFAULT 0,
                PRIMARY KEY (student_id, exam_id),
                FOREIGN KEY (student_id) REFERENCES users(id),
                FOREIGN KEY (exam_id) REFERENCES exams(id)
            )
        ''')

        # Proctoring logs
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS proctoring_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id INTEGER,
                exam_id INTEGER,
                violation_type TEXT,
                timestamp TEXT,
                details TEXT,
                FOREIGN KEY (student_id) REFERENCES users(id),
                FOREIGN KEY (exam_id) REFERENCES exams(id)
            )
        ''')
        
        self.conn.commit()
        print("✅ Database with user roles initialized")
        
        # Create default admin user
        self.create_default_users()
    
    def create_default_users(self):
        """Create default admin, teacher, and student accounts"""
        # Default Admin (username: admin, password: admin123)
        admin_hash = self.hash_password("admin123")
        self.cursor.execute('''
            INSERT OR IGNORE INTO users (username, password_hash, role, full_name, created_at) 
            VALUES (?, ?, ?, ?, ?)
        ''', ('admin', admin_hash, 'admin', 'System Administrator', datetime.now().isoformat()))
        
        # Default Teacher (username: teacher, password: teacher123)
        teacher_hash = self.hash_password("teacher123")
        self.cursor.execute('''
            INSERT OR IGNORE INTO users (username, password_hash, role, full_name, created_at) 
            VALUES (?, ?, ?, ?, ?)
        ''', ('teacher', teacher_hash, 'teacher', 'Demo Teacher', datetime.now().isoformat()))
        
        # Get teacher ID
        self.cursor.execute("SELECT id FROM users WHERE username = 'teacher'")
        teacher_id = self.cursor.fetchone()[0]
        self.cursor.execute("INSERT OR IGNORE INTO teachers (user_id, subject) VALUES (?, ?)", (teacher_id, 'Computer Science'))
        
        # Default Student (username: student, password: student123)
        student_hash = self.hash_password("student123")
        self.cursor.execute('''
            INSERT OR IGNORE INTO users (username, password_hash, role, full_name, created_at) 
            VALUES (?, ?, ?, ?, ?)
        ''', ('student', student_hash, 'student', 'Demo Student', datetime.now().isoformat()))
        
        # Get student ID
        self.cursor.execute("SELECT id FROM users WHERE username = 'student'")
        student_id = self.cursor.fetchone()[0]
        self.cursor.execute("INSERT OR IGNORE INTO students (user_id, roll_number, class, year) VALUES (?, ?, ?, ?)", 
                          (student_id, 'CS001', 'CSE-A', 4))
        
        self.conn.commit()
        print("✅ Default users created")
        
        # Add sample questions if none exist
        self.create_sample_questions()
    
    def create_sample_questions(self):
        """Create sample questions for demo"""
        self.cursor.execute("SELECT COUNT(*) FROM questions")
        if self.cursor.fetchone()[0] == 0:
            sample_questions = [
                ("What is machine learning?", "Computer Science", "AI", "easy", 10, 
                 "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed."),
                ("Explain the process of photosynthesis", "Biology", "Plant Biology", "medium", 10,
                 "Photosynthesis is the process by which green plants use sunlight to synthesize nutrients from carbon dioxide and water, producing oxygen as a byproduct."),
                ("What are the advantages of renewable energy?", "Environmental Science", "Energy", "medium", 10,
                 "Renewable energy sources offer advantages including reduced emissions, sustainability, and lower long-term costs."),
                ("Compare and contrast supervised and unsupervised learning", "Computer Science", "AI", "hard", 15,
                 "Supervised learning uses labeled data for prediction, while unsupervised learning finds patterns in unlabeled data for clustering."),
                ("Define Newton's first law of motion", "Physics", "Mechanics", "easy", 10,
                 "An object at rest stays at rest, and an object in motion stays in motion unless acted upon by an external force.")
            ]
            
            for q_text, subject, topic, difficulty, marks, ideal in sample_questions:
                self.cursor.execute('''
                    INSERT INTO questions (question_text, subject, topic, difficulty, marks, created_by, created_at, ideal_answer)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (q_text, subject, topic, difficulty, marks, 2, datetime.now().isoformat(), ideal))
            
            self.conn.commit()
            print("✅ Sample questions created")
            
            # Create a default assignment using these questions
            self.cursor.execute("SELECT id FROM questions LIMIT 5")
            q_ids = [row[0] for row in self.cursor.fetchall()]
            
            if q_ids:
                self.create_assignment(
                    teacher_id=2, # Default teacher ID
                    title="Demo Assignment: AI & Science",
                    description="A sample assignment to demonstrate the AI evaluation capabilities.",
                    question_ids=q_ids,
                    due_date=(datetime.now() + pd.Timedelta(days=7)).isoformat()
                )
                print("✅ Default assignment created")
    
    def hash_password(self, password: str) -> str:
        """Hash password for storage"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def verify_user(self, username: str, password: str) -> dict:
        """Verify user login"""
        password_hash = self.hash_password(password)
        
        self.cursor.execute('''
            SELECT id, username, role, full_name FROM users 
            WHERE username = ? AND password_hash = ?
        ''', (username, password_hash))
        
        user = self.cursor.fetchone()
        
        if user:
            user_id, username, role, full_name = user
            # Update last login
            self.cursor.execute('''
                UPDATE users SET last_login = ? WHERE id = ?
            ''', (datetime.now().isoformat(), user_id))
            self.conn.commit()
            
            return {
                'id': user_id,
                'username': username,
                'role': role,
                'full_name': full_name,
                'login_success': True
            }
        
        return {'login_success': False, 'error': 'Invalid credentials'}
    
    def get_questions_for_student(self, student_id: int, assignment_id: int = None):
        """Get questions available for a student"""
        if assignment_id:
            # Get questions from specific assignment
            self.cursor.execute('''
                SELECT q.id, q.question_text, q.subject, q.topic, q.difficulty, q.marks,
                       sa.answer_text, sa.final_score, sa.feedback
                FROM questions q
                JOIN assignment_questions aq ON q.id = aq.question_id
                LEFT JOIN student_answers sa ON q.id = sa.question_id AND sa.student_id = ?
                WHERE aq.assignment_id = ?
                ORDER BY aq.sequence
            ''', (student_id, assignment_id))
        else:
            # Get only questions created by teachers (not other roles)
            self.cursor.execute('''
                SELECT q.id, q.question_text, q.subject, q.topic, q.difficulty, q.marks,
                       sa.answer_text, sa.final_score, sa.feedback
                FROM questions q
                INNER JOIN users u ON q.created_by = u.id AND u.role = 'teacher'
                LEFT JOIN student_answers sa ON q.id = sa.question_id AND sa.student_id = ?
                ORDER BY q.id
            ''', (student_id,))
        
        questions = []
        for row in self.cursor.fetchall():
            questions.append({
                'id': row[0],
                'question_text': row[1],
                'subject': row[2],
                'topic': row[3],
                'difficulty': row[4],
                'marks': row[5],
                'student_answer': row[6],
                'score': row[7],
                'feedback': row[8],
                'answered': row[6] is not None
            })
        
        return questions
    
    def get_teacher_questions(self, teacher_id: int):
        """Get all questions created by a teacher"""
        self.cursor.execute('''
            SELECT q.id, q.question_text, q.subject, q.topic, q.difficulty, q.marks, 
                   q.created_at, q.ideal_answer,
                   COUNT(sa.id) as answer_count,
                   AVG(sa.final_score) as avg_score
            FROM questions q
            LEFT JOIN student_answers sa ON q.id = sa.question_id
            WHERE q.created_by = ?
            GROUP BY q.id
            ORDER BY q.created_at DESC
        ''', (teacher_id,))
        
        questions = []
        for row in self.cursor.fetchall():
            questions.append({
                'id': row[0],
                'question_text': row[1],
                'subject': row[2],
                'topic': row[3],
                'difficulty': row[4],
                'marks': row[5],
                'created_at': row[6],
                'ideal_answer': row[7],
                'answer_count': row[8] or 0,
                'avg_score': round(row[9], 2) if row[9] else 0
            })
        
        return questions
    
    def get_students_list(self):
        """Get all students with their submission stats"""
        self.cursor.execute('''
            SELECT u.id, u.username, u.full_name, u.created_at,
                   COUNT(sa.id) as submissions,
                   AVG(sa.final_score) as avg_score
            FROM users u
            LEFT JOIN student_answers sa ON u.id = sa.student_id
            WHERE u.role = 'student'
            GROUP BY u.id
            ORDER BY u.full_name
        ''')
        
        students = []
        for row in self.cursor.fetchall():
            students.append({
                'id': row[0],
                'username': row[1],
                'full_name': row[2],
                'created_at': row[3],
                'submissions': row[4] or 0,
                'avg_score': round(row[5], 1) if row[5] else 0,
                'status': 'Active'
            })
        
        return students
    
    def submit_student_answer(self, student_id: int, question_id: int, answer_text: str):
        """Submit and evaluate student answer — validates input first."""
        # Validate input
        is_valid, val_msg = self.validate_answer(answer_text)
        if not is_valid:
            return {'error': val_msg, 'success': False}

        # Get question details
        self.cursor.execute(
            'SELECT ideal_answer, question_text, marks, subject FROM questions WHERE id = ?',
            (question_id,)
        )
        question = self.cursor.fetchone()
        if not question:
            return {'error': 'Question not found', 'success': False}

        ideal_answer  = question[0]
        question_text = question[1]
        max_marks     = question[2] or 10
        subject       = question[3] or ''

        # Run AI evaluation
        from Advanced_Core.advanced_evaluator import AdvancedAnswerEvaluator
        evaluator = AdvancedAnswerEvaluator()
        result = evaluator.evaluate(
            question_text, ideal_answer, answer_text,
            subject=subject, max_marks=max_marks
        )

        # Grade info
        grade_info = get_grade_band(result['final_score'])

        # Store answer
        submitted_at = datetime.now().isoformat()
        self.cursor.execute('''
            INSERT INTO student_answers
            (student_id, question_id, answer_text, submitted_at, evaluated_at,
             final_score, confidence, feedback, layer_scores, is_evaluated)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            student_id, question_id, answer_text, submitted_at, submitted_at,
            result['final_score'], result['confidence'], result['feedback'],
            json.dumps(result['layer_scores']), 1
        ))
        self.conn.commit()

        return {
            'success':       True,
            'score':         result['final_score'],
            'grade':         grade_info['grade'],
            'grade_label':   grade_info['label'],
            'marks_obtained': result.get('marks_obtained', 0),
            'max_marks':      max_marks,
            'feedback':       result['feedback'],
            'confidence':     result['confidence'],
            'layer_scores':   result['layer_scores'],
            'submission_id':  self.cursor.lastrowid
        }
    
    def create_assignment(self, teacher_id: int, title: str, description: str, question_ids: list, due_date: str = None):
        """Create new assignment"""
        created_at = datetime.now().isoformat()
        
        self.cursor.execute('''
            INSERT INTO assignments (title, description, teacher_id, created_at, due_date)
            VALUES (?, ?, ?, ?, ?)
        ''', (title, description, teacher_id, created_at, due_date))
        
        assignment_id = self.cursor.lastrowid
        
        # Add questions to assignment
        for seq, q_id in enumerate(question_ids, 1):
            self.cursor.execute('''
                INSERT INTO assignment_questions (assignment_id, question_id, sequence)
                VALUES (?, ?, ?)
            ''', (assignment_id, q_id, seq))
        
        self.conn.commit()
        
        return assignment_id
    
    def get_student_assignments(self, student_id: int):
        """Get assignments for a student"""
        self.cursor.execute('''
            SELECT a.id, a.title, a.description, a.created_at, a.due_date,
                   u.full_name as teacher_name,
                   sa.status, sa.total_score
            FROM assignments a
            JOIN users u ON a.teacher_id = u.id
            LEFT JOIN student_assignments sa ON a.id = sa.assignment_id AND sa.student_id = ?
            WHERE a.is_active = 1
            ORDER BY a.due_date ASC
        ''', (student_id,))
        
        assignments = []
        for row in self.cursor.fetchall():
            assignments.append({
                'id': row[0],
                'title': row[1],
                'description': row[2],
                'created_at': row[3],
                'due_date': row[4],
                'teacher_name': row[5],
                'status': row[6] or 'not_started',
                'score': row[7]
            })
        
        return assignments
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()

    def get_question(self, question_id: int):
        """Get question details by ID"""
        self.cursor.execute('''
            SELECT * FROM questions WHERE id = ?
        ''', (question_id,))
        row = self.cursor.fetchone()
        
        if not row:
            return None
            
        # Get column names
        cols = [description[0] for description in self.cursor.description]
        return dict(zip(cols, row))

    def create_question(self, teacher_id: int, question_text: str, subject: str, 
                   topic: str, difficulty: str, marks: int, ideal_answer: str, 
                   keywords: str = None):
        """Create a new question"""
        created_at = datetime.now().isoformat()
        
        self.cursor.execute('''
            INSERT INTO questions 
            (question_text, subject, topic, difficulty, marks, created_by, created_at, ideal_answer, keywords)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (question_text, subject, topic, difficulty, marks, teacher_id, created_at, ideal_answer, keywords))
        
        self.conn.commit()
        self.conn.commit()
        
        # Save to Real Dataset (CSV)
        self._save_to_dataset("questions.csv", {
            "id": self.cursor.lastrowid,
            "question_text": question_text,
            "subject": subject,
            "topic": topic,
            "difficulty": difficulty,
            "marks": marks,
            "ideal_answer": ideal_answer,
            "keywords": keywords,
            "created_at": created_at
        })
        
        return self.cursor.lastrowid
    
    def delete_question(self, question_id: int):
        """Delete a question and all its related data (answers, exam links)"""
        # 1. Delete student answers for this question
        self.cursor.execute("DELETE FROM student_answers WHERE question_id = ?", (question_id,))
        
        # 2. Delete from exam questions
        self.cursor.execute("DELETE FROM exam_questions WHERE question_id = ?", (question_id,))
        
        # 3. Delete from assignment questions
        self.cursor.execute("DELETE FROM assignment_questions WHERE question_id = ?", (question_id,))
        
        # 4. Delete the question itself
        self.cursor.execute("DELETE FROM questions WHERE id = ?", (question_id,))
        
        self.conn.commit()
        return True

    def submit_answer(self, student_id: int, question_id: int, answer_text: str,
                     evaluation_result: dict):
        """
        Submit and save student answer with detailed AI scores.
        Always keeps latest attempt (upsert pattern).
        """
        # Validate input
        is_valid, val_msg = self.validate_answer(answer_text)
        if not is_valid:
            return False

        submitted_at  = datetime.now().isoformat()
        final_score   = evaluation_result.get('final_score', 0)
        confidence    = evaluation_result.get('confidence', 0)
        feedback      = evaluation_result.get('feedback', '')
        layer_scores  = json.dumps(evaluation_result.get('layer_scores', {}))
        grade         = evaluation_result.get('grade', '')
        marks_obtained = evaluation_result.get('marks_obtained', 0)

        # Check if already answered — update if so (keep best attempt)
        self.cursor.execute(
            'SELECT id, final_score FROM student_answers WHERE student_id=? AND question_id=?',
            (student_id, question_id)
        )
        existing = self.cursor.fetchone()

        if existing:
            # Only overwrite if new score is better (or equal)
            existing_score = existing[1] or 0
            if final_score >= existing_score:
                self.cursor.execute('''
                    UPDATE student_answers
                    SET answer_text=?, final_score=?, confidence=?, feedback=?,
                        layer_scores=?, submitted_at=?, is_evaluated=1
                    WHERE id=?
                ''', (answer_text, final_score, confidence, feedback,
                      layer_scores, submitted_at, existing[0]))
        else:
            self.cursor.execute('''
                INSERT INTO student_answers
                (student_id, question_id, answer_text, final_score, confidence, feedback,
                 layer_scores, submitted_at, is_evaluated)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1)
            ''', (student_id, question_id, answer_text, final_score, confidence,
                  feedback, layer_scores, submitted_at))

        self.conn.commit()

        # Persist to CSV dataset
        self._save_to_dataset('student_submissions.csv', {
            'student_id':    student_id,
            'question_id':   question_id,
            'answer_text':   answer_text,
            'final_score':   final_score,
            'grade':         grade,
            'marks_obtained': marks_obtained,
            'confidence':    confidence,
            'feedback':      feedback,
            'submitted_at':  submitted_at
        })

        return True

    def get_class_analytics(self, teacher_id: int):
        """Get aggregated analytics for a teacher's classes"""
        # Overall stats
        self.cursor.execute('''
            SELECT COUNT(DISTINCT sa.student_id) as active_students,
                   COUNT(sa.id) as total_submissions,
                   AVG(sa.final_score) as avg_score
            FROM questions q
            JOIN student_answers sa ON q.id = sa.question_id
            WHERE q.created_by = ?
        ''', (teacher_id,))
        stats = self.cursor.fetchone()
        
        # Topic-wise performance
        self.cursor.execute('''
            SELECT q.topic, ROUND(AVG(sa.final_score), 1) as score, COUNT(sa.id) as count
            FROM questions q
            JOIN student_answers sa ON q.id = sa.question_id
            WHERE q.created_by = ?
            GROUP BY q.topic
            ORDER BY score DESC
        ''', (teacher_id,))
        topic_rows = self.cursor.fetchall()
        topic_performance = [{'topic': r[0] or 'General', 'score': r[1] or 0, 'count': r[2]} for r in topic_rows]
        
        # If no topics yet, provide sample data for demo
        if not topic_performance:
            topic_performance = [
                {'topic': 'AI', 'score': 78.5, 'count': 3},
                {'topic': 'Plant Biology', 'score': 72.0, 'count': 2},
                {'topic': 'Energy', 'score': 68.5, 'count': 2},
                {'topic': 'Mechanics', 'score': 82.0, 'count': 1},
                {'topic': 'Programming', 'score': 75.0, 'count': 2}
            ]
        
        # Layer-wise averages from stored JSON
        self.cursor.execute('''
            SELECT sa.layer_scores
            FROM student_answers sa
            JOIN questions q ON sa.question_id = q.id
            WHERE q.created_by = ? AND sa.layer_scores IS NOT NULL
        ''', (teacher_id,))
        
        layer_rows = self.cursor.fetchall()
        layer_avgs = {'conceptual': 0, 'semantic': 0, 'structural': 0, 'completeness': 0}
        
        if layer_rows:
            count = 0
            for row in layer_rows:
                try:
                    scores = json.loads(row[0])
                    for key in layer_avgs:
                        layer_avgs[key] += scores.get(key, 0)
                    count += 1
                except:
                    pass
            if count > 0:
                for key in layer_avgs:
                    layer_avgs[key] = round(layer_avgs[key] / count, 1)
        
        return {
            'active_students': stats[0] or 0,
            'total_submissions': stats[1] or 0,
            'avg_score': round(stats[2] or 0, 1),
            'topic_performance': topic_performance,
            'layer_averages': layer_avgs
        }

    def get_top_students(self, teacher_id: int, limit: int = 5):
        """Get top performing students"""
        self.cursor.execute('''
            SELECT u.full_name, AVG(sa.final_score) as avg_score, COUNT(sa.id) as answers_count
            FROM students s
            JOIN users u ON s.user_id = u.id
            JOIN student_answers sa ON s.user_id = sa.student_id
            JOIN questions q ON sa.question_id = q.id
            WHERE q.created_by = ?
            GROUP BY u.id
            ORDER BY avg_score DESC
            LIMIT ?
        ''', (teacher_id, limit))
        
        students = []
        for row in self.cursor.fetchall():
            students.append({
                'name': row[0],
                'avg_score': round(row[1], 1),
                'answers_count': row[2]
            })
        return students

    def get_student_submissions(self, teacher_id: int):
        """Get all student submissions for questions created by this teacher"""
        self.cursor.execute('''
            SELECT sa.id, u.full_name, q.question_text, sa.answer_text, sa.final_score, 
                   sa.confidence, sa.layer_scores, sa.submitted_at, q.topic
            FROM student_answers sa
            JOIN users u ON sa.student_id = u.id
            JOIN questions q ON sa.question_id = q.id
            WHERE q.created_by = ?
            ORDER BY sa.submitted_at DESC
        ''', (teacher_id,))
        
        submissions = []
        for row in self.cursor.fetchall():
            # Parse layer scores safely
            try:
                layer_scores = json.loads(row[6]) if row[6] else {}
            except:
                layer_scores = {}

            submissions.append({
                'id': row[0],
                'student_name': row[1],
                'question': row[2],
                'answer': row[3],
                'score': row[4],
                'confidence': row[5],
                'layer_scores': layer_scores,
                'submitted_at': row[7],
                'topic': row[8]
            })
        
        return submissions

    def get_student_analytics(self, student_id: int):
        """Get detailed analytics for a student including NLP layer scores"""
        # Overall progress
        self.cursor.execute('''
            SELECT AVG(final_score), COUNT(*) 
            FROM student_answers 
            WHERE student_id = ?
        ''', (student_id,))
        stats = self.cursor.fetchone()
        
        # Detailed layer scores (Parsing JSON needed in app, here just fetch raw)
        self.cursor.execute('''
            SELECT q.subject, sa.final_score, sa.layer_scores, sa.submitted_at
            FROM student_answers sa
            JOIN questions q ON sa.question_id = q.id
            WHERE sa.student_id = ?
            ORDER BY sa.submitted_at ASC
        ''', (student_id,))
        
        history = []
        for row in self.cursor.fetchall():
            try:
                layers = json.loads(row[2]) if row[2] else {}
            except:
                layers = {}
                
            history.append({
                'subject': row[0],
                'score': row[1],
                'layers': layers,
                'date': row[3]
            })
            
        return {
            'avg_score': round(stats[0] or 0, 1),
            'total_answers': stats[1] or 0,
            'history': history
        }
    
    def get_student_answers(self, student_id: int):
        """Get all answers by a student"""
        self.cursor.execute('''
            SELECT q.question_text, sa.answer_text, sa.final_score, sa.feedback, 
                   sa.submitted_at, q.subject, q.ideal_answer
            FROM student_answers sa
            JOIN questions q ON sa.question_id = q.id
            WHERE sa.student_id = ?
            ORDER BY sa.submitted_at DESC
        ''', (student_id,))
        
        answers = []
        for row in self.cursor.fetchall():
            answers.append({
                'question': row[0],
                'answer': row[1],
                'score': row[2],
                'feedback': row[3],
                'submitted_at': row[4],
                'subject': row[5],
                'ideal_answer': row[6]
            })
        
        return answers

    def create_exam(self, teacher_id: int, title: str, description: str, 
                   duration_minutes: int, start_time: str, end_time: str, 
                   question_ids: list, strict_mode: int = 1):
        """Create new exam"""
        self.cursor.execute('''
            INSERT INTO exams (title, description, teacher_id, duration_minutes, start_time, end_time, strict_mode)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (title, description, teacher_id, duration_minutes, start_time, end_time, strict_mode))
        
        exam_id = self.cursor.lastrowid
        
        # Add questions
        for seq, q_id in enumerate(question_ids, 1):
            self.cursor.execute('''
                INSERT INTO exam_questions (exam_id, question_id, sequence)
                VALUES (?, ?, ?)
            ''', (exam_id, q_id, seq))
        
        self.conn.commit()
        return exam_id

    def get_student_exams(self, student_id: int):
        """Get exams for a student"""
        # Get all active exams
        self.cursor.execute('''
            SELECT e.id, e.title, e.description, e.start_time, e.end_time, e.duration_minutes,
                   u.full_name as teacher_name,
                   se.status, se.total_score, se.proctoring_score
            FROM exams e
            JOIN users u ON e.teacher_id = u.id
            LEFT JOIN student_exams se ON e.id = se.exam_id AND se.student_id = ?
            WHERE e.is_active = 1
            ORDER BY e.start_time DESC
        ''', (student_id,))
        
        exams = []
        now = datetime.now()
        
        for row in self.cursor.fetchall():
            start_time = datetime.fromisoformat(row[3]) if row[3] else datetime.min
            end_time = datetime.fromisoformat(row[4]) if row[4] else datetime.max
            
            # Determine status if not started
            status = row[7] or 'not_started'
            is_open = start_time <= now <= end_time
            
            exams.append({
                'id': row[0],
                'title': row[1],
                'description': row[2],
                'start_time': row[3],
                'end_time': row[4],
                'duration_minutes': row[5],
                'teacher_name': row[6],
                'status': status,
                'score': row[8],
                'proctoring_score': row[9],
                'is_open': is_open
            })
        
        return exams

    def start_exam(self, student_id: int, exam_id: int):
        """Start an exam for a student"""
        started_at = datetime.now().isoformat()
        
        self.cursor.execute('''
            INSERT OR IGNORE INTO student_exams (student_id, exam_id, started_at, status)
            VALUES (?, ?, ?, 'in_progress')
        ''', (student_id, exam_id, started_at))
        
        self.conn.commit()
        return True

    def log_proctoring_violation(self, student_id: int, exam_id: int, violation_type: str):
        """Log a proctoring violation"""
        timestamp = datetime.now().isoformat()
        
        self.cursor.execute('''
            INSERT INTO proctoring_logs (student_id, exam_id, violation_type, timestamp)
            VALUES (?, ?, ?, ?)
        ''', (student_id, exam_id, violation_type, timestamp))
        
        # Update proctoring score
        self.cursor.execute('''
            UPDATE student_exams 
            SET proctoring_score = MAX(0, proctoring_score - 5),
                violation_count = violation_count + 1
            WHERE student_id = ? AND exam_id = ?
        ''', (student_id, exam_id))
        
        self.conn.commit()
        return True

    def get_exam_questions(self, exam_id: int):
        """Get questions for an exam"""
        self.cursor.execute('''
            SELECT q.id, q.question_text, q.marks, q.difficulty, q.subject
            FROM questions q
            JOIN exam_questions eq ON q.id = eq.question_id
            WHERE eq.exam_id = ?
            ORDER BY eq.sequence
        ''', (exam_id,))
        
        questions = []
        for row in self.cursor.fetchall():
            questions.append({
                'id': row[0],
                'question_text': row[1],
                'marks': row[2],
                'difficulty': row[3],
                'subject': row[4]
            })
        return questions

    def get_exam_details(self, exam_id: int):
        """Get exam details"""
        self.cursor.execute('''
            SELECT id, title, description, duration_minutes, strict_mode
            FROM exams
            WHERE id = ?
        ''', (exam_id,))
        
        row = self.cursor.fetchone()
        if row:
            return {
                'id': row[0],
                'title': row[1],
                'description': row[2],
                'duration_minutes': row[3],
                'strict_mode': row[4]
            }
        return None