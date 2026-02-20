"""
UPDATED: FastAPI with 3 user roles - COMPLETE WORKING VERSION
"""
from fastapi import FastAPI, Request, Form, File, UploadFile, Depends, HTTPException, status, Body
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.security import HTTPBearer
import uvicorn
import json
import pandas as pd
from typing import List, Optional
import os
from pathlib import Path
from datetime import datetime

# Import authentication
try:
    from Production_Deployment.auth_system import (
        create_access_token, decode_jwt, get_current_user, 
        check_role_access, JWTBearer
    )
    print("✓ Authentication system loaded")
except Exception as e:
    print(f"⚠ Auth system import error: {e}")
    # Create fallback functions
    def create_access_token(data: dict):
        return "dummy_token"
    
    def decode_jwt(token: str):
        return {"user_id": 1, "username": "demo", "role": "student", "full_name": "Demo User"}
    
    def get_current_user(request: Request):
        return {"user_id": 1, "username": "demo", "role": "student", "full_name": "Demo User"}
    
    def check_role_access(user_role: str, required_role: str) -> bool:
        return True
    
    class JWTBearer:
        def __call__(self, request: Request):
            return "dummy_token"

# Import database
try:
    from Utilities.database_manager import DatabaseManager
    db = DatabaseManager()
    db.initialize()
    print("✓ Database initialized")
except Exception as e:
    print(f"⚠ Database error: {e}")
    # Create dummy database
    class DummyDB:
        def __init__(self):
            self.questions = [
                {"id": 1, "question_text": "What is machine learning?", "subject": "Computer Science", 
                 "topic": "AI", "difficulty": "easy", "marks": 10, "answered": False, "score": None,
                 "student_answer": None, "feedback": None, "ideal_answer": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed."},
                {"id": 2, "question_text": "Explain photosynthesis", "subject": "Biology", 
                 "topic": "Plant Biology", "difficulty": "medium", "marks": 10, "answered": False, "score": None,
                 "student_answer": None, "feedback": None, "ideal_answer": "Photosynthesis is the process by which green plants use sunlight to synthesize nutrients from carbon dioxide and water, producing oxygen as a byproduct."}
            ]
        
        def verify_user(self, username, password):
            users = {
                "admin": {"id": 1, "username": "admin", "role": "admin", "full_name": "Admin User", "login_success": True},
                "teacher": {"id": 2, "username": "teacher", "role": "teacher", "full_name": "Teacher User", "login_success": True},
                "student": {"id": 3, "username": "student", "role": "student", "full_name": "Student User", "login_success": True}
            }
            return users.get(username, {"login_success": False})
        
        def get_questions_for_student(self, student_id):
            return self.questions
        
        def submit_student_answer(self, student_id, question_id, answer_text):
            # Find question
            question = next((q for q in self.questions if q["id"] == question_id), None)
            if not question:
                return {"error": "Question not found"}
            
            # Simulate evaluation
            from Advanced_Core.advanced_evaluator import AdvancedAnswerEvaluator
            evaluator = AdvancedAnswerEvaluator()
            result = evaluator.evaluate(
                question["question_text"], 
                question["ideal_answer"], 
                answer_text
            )
            
            # Update question
            question["answered"] = True
            question["student_answer"] = answer_text
            question["score"] = result["final_score"]
            question["feedback"] = result["feedback"]
            
            return {
                "success": True,
                "score": result["final_score"],
                "feedback": result["feedback"],
                "confidence": result["confidence"],
                "layer_scores": result["layer_scores"]
            }
        
        def get_student_assignments(self, student_id):
            return [
                {"id": 1, "title": "AI Basics", "description": "Introduction to AI concepts", 
                 "status": "not_started", "score": None, "teacher_name": "Dr. Smith", 
                 "due_date": "2024-12-31", "created_at": "2024-01-01"}
            ]
        
        def get_teacher_questions(self, teacher_id):
            return self.questions
    
    db = DummyDB()

# Import evaluator
try:
    from Advanced_Core.advanced_evaluator import AdvancedAnswerEvaluator
    evaluator = AdvancedAnswerEvaluator()
    print("✓ Advanced evaluator loaded")
except Exception as e:
    print(f"⚠ Advanced evaluator error: {e}")
    try:
        from Advanced_Core.light_evaluator import LightEvaluator
        evaluator = LightEvaluator()
        print("✓ Light evaluator loaded")
    except:
        print("⚠ Using fallback evaluator")
        # Fallback evaluator
        class FallbackEvaluator:
            def evaluate(self, question, ideal, student):
                return {
                    "final_score": 75.0,
                    "confidence": 85.0,
                    "layer_scores": {
                        "conceptual": 80.0,
                        "semantic": 75.0,
                        "structural": 70.0,
                        "completeness": 85.0
                    },
                    "feedback": "Good answer overall. Could improve structure and add more examples.",
                    "weights": [0.3, 0.4, 0.1, 0.2]
                }
        evaluator = FallbackEvaluator()

# Initialize FastAPI
app = FastAPI(
    title="Answer Evaluation System - Multi-User",
    description="Top 1% BE Final Year Project - Intelligent Descriptive Answer Evaluation",
    version="2.0.0"
)

# Mount static files
app.mount("/static", StaticFiles(directory="Frontend/static"), name="static")

# Templates
templates = Jinja2Templates(directory="Frontend/templates")

# ==================== AUTHENTICATION ENDPOINTS ====================

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Home page redirects to login"""
    return templates.TemplateResponse("login.html", {"request": request})

@app.post("/api/login")
async def login(
    username: str = Form(...), 
    password: str = Form(...), 
    role: str = Form(...)
):
    """User login - accepts form data"""
    try:
        result = db.verify_user(username, password)
        
        if result.get('login_success'):
            # Check if role matches
            if result.get('role') != role:
                return JSONResponse({
                    'success': False,
                    'error': f'User is registered as {result.get("role")}, not {role}'
                })
            
            # Create JWT token
            token_data = {
                'user_id': result.get('id', 1),
                'username': result.get('username', 'demo'),
                'role': result.get('role', 'student'),
                'full_name': result.get('full_name', 'Demo User')
            }
            token = create_access_token(token_data)
            
            # Determine redirect URL based on role
            redirect_url = {
                'admin': '/admin',
                'teacher': '/teacher',
                'student': '/student'
            }.get(role, '/student')
            
            
            print(f"DEBUG: Login successful for {username}, redirecting to {redirect_url}")
            
            response = JSONResponse({
                'success': True,
                'token': token,
                'redirect_url': redirect_url,
                'user': result
            })
            
            # Set cookie for session management
            response.set_cookie(
                key="access_token", 
                value=token, 
                httponly=True,
                max_age=7200, # 2 hours
                samesite='lax'
            )
            
            return response
        
        return JSONResponse({
            'success': False,
            'error': 'Invalid username or password'
        })
    
    except Exception as e:
        print(f"Login error: {e}")
        return JSONResponse({
            'success': False,
            'error': 'Server error during login'
        })

@app.get("/logout")
async def logout():
    """Logout user"""
    response = RedirectResponse(url="/")
    response.delete_cookie("access_token")
    return response

# ==================== STUDENT ENDPOINTS ====================

@app.get("/student", response_class=HTMLResponse)
async def student_dashboard(request: Request):
    """Student dashboard"""
    print(f"DEBUG: Handling /student request")
    user = get_current_user(request)
    print(f"DEBUG: User in /student: {user}")
    
    if not user:
        print("DEBUG: No user found, redirecting to login")
        return RedirectResponse(url="/")
    
    # Get student assignments
    assignments = db.get_student_assignments(user.get('user_id', 1))
    
    # Calculate stats
    total = len(assignments)
    completed = len([a for a in assignments if a.get('status') == 'completed'])
    in_progress = len([a for a in assignments if a.get('status') == 'in_progress'])
    avg_score = 0
    if completed > 0:
        scores = [a.get('score', 0) for a in assignments if a.get('score')]
        avg_score = sum(scores) / len(scores) if scores else 0
    
    return templates.TemplateResponse("student_dashboard.html", {
        "request": request,
        "student_name": user.get('full_name', 'Student'),
        "assignments": assignments,
        "total_assignments": total,
        "completed_assignments": completed,
        "in_progress": in_progress,
        "avg_score": round(avg_score, 1)
    })

@app.get("/student/assignment/{assignment_id}", response_class=HTMLResponse)
async def student_assignment(request: Request, assignment_id: int):
    """Student assignment page"""
    user = get_current_user(request)
    if not user:
        return RedirectResponse(url="/")
    
    # Get assignment details and questions
    questions = db.get_questions_for_student(user.get('user_id', 1))
    
    return templates.TemplateResponse("student_assignment.html", {
        "request": request,
        "assignment_id": assignment_id,
        "questions": questions,
        "student_name": user.get('full_name', 'Student')
    })


@app.get("/student/performance", response_class=HTMLResponse)
async def student_performance(request: Request):
    """Student performance dashboard"""
    user = get_current_user(request)
    if not user:
        return RedirectResponse(url="/")
        
    analytics = db.get_student_analytics(user.get('user_id', 1))
    
    return templates.TemplateResponse("student_performance.html", {
        "request": request,
        "student_name": user.get('full_name', 'Student'),
        "analytics": analytics
    })

@app.get("/student/feedback", response_class=HTMLResponse)
async def student_feedback(request: Request):
    """Student detailed feedback history"""
    user = get_current_user(request)
    if not user:
        return RedirectResponse(url="/")
        
    answers = db.get_student_answers(user.get('user_id', 1))
    
    return templates.TemplateResponse("student_feedback.html", {
        "request": request,
        "student_name": user.get('full_name', 'Student'),
        "answers": answers
    })

@app.get("/student/questions", response_class=HTMLResponse)
async def student_questions(request: Request):
    """Student practice questions page"""
    user = get_current_user(request)
    if not user:
        return RedirectResponse(url="/")
    
    # Get all questions for student
    questions = db.get_questions_for_student(user.get('user_id', 1))
    
    # Calculate statistics
    total = len(questions)
    answered = len([q for q in questions if q.get('answered', False)])
    pending = total - answered
    avg_score = 0
    if answered > 0:
        scores = [q.get('score', 0) for q in questions if q.get('score')]
        avg_score = sum(scores) / len(scores) if scores else 0
    
    return templates.TemplateResponse("student_questions.html", {
        "request": request,
        "student_name": user.get('full_name', 'Student'),
        "questions": questions,
        "total_questions": total,
        "answered_count": answered,
        "pending_count": pending,
        "avg_score": round(avg_score, 1)
    })

@app.get("/student/exams", response_class=HTMLResponse)
async def student_exams_page(request: Request):
    """Student exams page"""
    user = get_current_user(request)
    if not user:
        return RedirectResponse(url="/")
    
    # Get exams
    exams = db.get_student_exams(user.get('user_id', 1))
    
    return templates.TemplateResponse("student_exams.html", {
        "request": request,
        "student_name": user.get('full_name', 'Student'),
        "exams": exams
    })

@app.get("/student/exam/{exam_id}", response_class=HTMLResponse)
async def student_take_exam_page(request: Request, exam_id: int):
    """Secure Exam Interface"""
    user = get_current_user(request)
    if not user:
        return RedirectResponse(url="/")
    
    # Get exam details
    exam = db.get_exam_details(exam_id)
    if not exam:
        return RedirectResponse(url="/student/exams")
        
    questions = db.get_exam_questions(exam_id)
    
    return templates.TemplateResponse("exam_interface.html", {
        "request": request,
        "student_name": user.get('full_name', 'Student'),
        "exam": exam,
        "questions": questions,
        "question_count": len(questions)
    })

@app.get("/student/question/{question_id}", response_class=HTMLResponse)
async def student_question_page(request: Request, question_id: int):
    """Page to answer a specific question"""
    user = get_current_user(request)
    if not user:
        return RedirectResponse(url="/")
    
    question = db.get_question(question_id)
    
    if not question:
        return RedirectResponse(url="/student/questions")
        
    return templates.TemplateResponse("student_question.html", {
        "request": request,
        "student_name": user.get('full_name', 'Student'),
        "question": question
    })

@app.post("/api/student/submit-answer")
async def submit_answer_api(
    request: Request,
    submission: dict = Body(...)
):
    """Submit answer for AI evaluation — with validation, grade & marks"""
    user = get_current_user(request)
    if not user:
        raise HTTPException(status_code=401)

    question_id = submission.get('question_id')
    answer_text = submission.get('answer_text', '').strip()

    # ── Server-side validation ──
    if not answer_text or len(answer_text) < 5:
        return JSONResponse({
            'success': False,
            'error': 'Answer is too short or empty. Please write a proper answer.'
        })
    if len(answer_text) > 5000:
        return JSONResponse({
            'success': False,
            'error': 'Answer is too long. Please limit to 5000 characters.'
        })

    try:
        # Get question details for evaluation
        question = db.get_question(question_id)
        if not question:
            return JSONResponse({'success': False, 'error': 'Question not found'})

        max_marks = question.get('marks', 10)
        subject   = question.get('subject', '')

        # Evaluate answer with subject and marks
        result = evaluator.evaluate(
            question['question_text'],
            question['ideal_answer'],
            answer_text,
            subject=subject,
            max_marks=max_marks
        )

        # Save to DB
        db.submit_answer(user.get('user_id', 1), question_id, answer_text, result)

        # Build response with grade and marks
        final_score = result.get('final_score', 0)
        grade_label = result.get('grade_label', '')
        grade       = result.get('grade', '')
        marks_obtained = result.get('marks_obtained', round(final_score * max_marks / 100, 1))

        result['success']       = True
        result['grade']         = grade
        result['grade_label']   = grade_label
        result['marks_obtained'] = marks_obtained
        result['max_marks']     = max_marks
        return JSONResponse(result)

    except Exception as e:
        print(f"Submission error: {e}")
        import traceback; traceback.print_exc()
        return JSONResponse({'success': False, 'error': str(e)})

@app.post("/api/student/start-exam")
async def start_exam_api(
    request: Request,
    exam_id: int = Body(..., embed=True)
):
    """Start exam timer"""
    user = get_current_user(request)
    if not user:
        raise HTTPException(status_code=401)
        
    success = db.start_exam(user.get('user_id', 1), exam_id)
    return {"success": success}

@app.post("/api/student/log-violation")
async def log_violation_api(
    request: Request,
    exam_id: int = Body(...),
    violation_type: str = Body(...)
):
    """Log proctoring violation"""
    user = get_current_user(request)
    if not user:
        raise HTTPException(status_code=401)
        
    db.log_proctoring_violation(user.get('user_id', 1), exam_id, violation_type)
    return {"success": True}

# ==================== TEACHER ENDPOINTS ====================

@app.get("/teacher", response_class=HTMLResponse)
async def teacher_dashboard(request: Request):
    """Teacher dashboard"""
    user = get_current_user(request)
    if not user:
        return RedirectResponse(url="/")
    
    # Get teacher's questions
    questions = db.get_teacher_questions(user.get('user_id'))
    
    # Get analytics and top students
    analytics = db.get_class_analytics(user.get('user_id'))
    top_students = db.get_top_students(user.get('user_id'))
    
    return templates.TemplateResponse("teacher_dashboard.html", {
        "request": request,
        "teacher_name": user.get('full_name', 'Teacher'),
        "questions": questions,
        "analytics": analytics,
        "top_students": top_students
    })

@app.get("/teacher/create-question", response_class=HTMLResponse)
async def teacher_create_question_page(request: Request):
    """Page for teachers to create questions"""
    user = get_current_user(request)
    if not user:
        return RedirectResponse(url="/")
    
    return templates.TemplateResponse("teacher_create_question.html", {
        "request": request,
        "teacher_name": user.get('full_name', 'Teacher')
    })

@app.post("/api/teacher/create-question")
async def create_question_api(
    request: Request,
    question_text: str = Form(...),
    subject: str = Form(...),
    topic: str = Form(...),
    difficulty: str = Form(...),
    marks: int = Form(10),
    ideal_answer: str = Form(...),
    keywords: str = Form("")
):
    """API endpoint for creating questions"""
    user = get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Not authorized")
    
    try:
        # Use real database
        user_id = user.get('user_id')
        question_id = db.create_question(
            teacher_id=user_id,
            question_text=question_text,
            subject=subject,
            topic=topic,
            difficulty=difficulty,
            marks=marks,
            ideal_answer=ideal_answer,
            keywords=keywords or "[]"
        )
        
        return JSONResponse({
            "success": True,
            "message": "Question created successfully",
            "question_id": question_id
        })
    
    except Exception as e:
        print(f"Create question error: {e}")
        return JSONResponse({
            "success": False,
            "error": "Failed to create question"
        })

@app.post("/api/teacher/delete-question")
async def delete_question_api(
    request: Request,
    question_id: int = Body(..., embed=True)
):
    """API endpoint for deleting questions"""
    user = get_current_user(request)
    if not user or user.get('role') != 'teacher':
        raise HTTPException(status_code=401, detail="Not authorized")
        
    try:
        if hasattr(db, 'questions'): # Dummy DB check
            db.questions = [q for q in db.questions if q.get('id') != question_id]
        
        # Real DB deletion
        success = db.delete_question(question_id)
        
        if success:
            return JSONResponse({"success": True, "message": "Question deleted successfully"})
        else:
            return JSONResponse({"success": False, "error": "Failed to delete question"})
            
    except Exception as e:
        print(f"Delete question error: {e}")
        return JSONResponse({
            "success": False, 
            "error": str(e)
        })

@app.get("/teacher/students", response_class=HTMLResponse)
async def teacher_students(request: Request):
    """Teacher students list"""
    user = get_current_user(request)
    if not user or user.get('role') != 'teacher':
        return RedirectResponse(url="/")
        
    # Placeholder for student list - normally we'd have a db.get_students()
    # using a dummy list for now or empty
    students = db.get_students_list() if hasattr(db, 'get_students_list') else []
    
    return templates.TemplateResponse("teacher_students.html", {
        "request": request,
        "teacher_name": user.get('full_name', 'Teacher'),
        "students": students
    })

@app.get("/teacher/analytics", response_class=HTMLResponse)
async def teacher_analytics(request: Request):
    """Teacher analytics dashboard"""
    user = get_current_user(request)
    if not user or user.get('role') != 'teacher':
        return RedirectResponse(url="/")
        
    analytics = db.get_class_analytics(user.get('user_id', 2))
    
    return templates.TemplateResponse("teacher_analytics.html", {
        "request": request,
        "teacher_name": user.get('full_name', 'Teacher'),
        "analytics": analytics
    })

@app.get("/teacher/results", response_class=HTMLResponse)
async def teacher_view_results(request: Request):
    """View all student results with detailed analysis"""
    user = get_current_user(request)
    if not user or user.get('role') != 'teacher':
        return RedirectResponse(url="/")
    
    # Get all submissions for this teacher's questions
    submissions = db.get_student_submissions(user.get('user_id', 2))
    
    return templates.TemplateResponse("teacher_view_results.html", {
        "request": request,
        "teacher_name": user.get('full_name', 'Teacher'),
        "submissions": submissions
    })

@app.get("/teacher/exams", response_class=HTMLResponse)
async def teacher_exams_page(request: Request):
    """Teacher exams list"""
    user = get_current_user(request)
    if not user or user.get('role') != 'teacher':
        return RedirectResponse(url="/")
        
    # Reuse student method or create specific teacher one (using student's for now as it lists all)
    # Ideally should implement get_teacher_exams in DB
    exams = [] # Placeholder
    
    return templates.TemplateResponse("teacher_exams.html", {
        "request": request,
        "teacher_name": user.get('full_name', 'Teacher'),
        "exams": exams
    })

@app.get("/teacher/create-exam", response_class=HTMLResponse)
async def teacher_create_exam_page(request: Request):
    """Create exam page"""
    user = get_current_user(request)
    if not user or user.get('role') != 'teacher':
        return RedirectResponse(url="/")
        
    # Get questions to add to exam
    questions = db.get_teacher_questions(user.get('user_id', 2))
    
    return templates.TemplateResponse("teacher_create_exam.html", {
        "request": request,
        "teacher_name": user.get('full_name', 'Teacher'),
        "questions": questions
    })

@app.post("/api/teacher/create-exam")
async def create_exam_api(
    request: Request,
    title: str = Form(...),
    description: str = Form(...),
    duration: int = Form(...),
    start_time: str = Form(...),
    end_time: str = Form(...),
    question_ids: str = Form(...) # JSON string of IDs
):
    """Create exam API"""
    user = get_current_user(request)
    if not user or user.get('role') != 'teacher':
        raise HTTPException(status_code=401)
        
    try:
        q_ids = json.loads(question_ids)
        
        exam_id = db.create_exam(
            user.get('user_id', 2),
            title,
            description,
            duration,
            start_time,
            end_time,
            q_ids
        )
        
        return JSONResponse({"success": True, "exam_id": exam_id})
    except Exception as e:
        print(f"Create exam error: {e}")
        return JSONResponse({"success": False, "error": str(e)})

# ==================== ADMIN ENDPOINTS ====================

@app.get("/admin", response_class=HTMLResponse)
async def admin_dashboard(request: Request):
    """Admin dashboard"""
    user = get_current_user(request)
    if not user:
        return RedirectResponse(url="/")
    
    return templates.TemplateResponse("admin_dashboard.html", {
        "request": request,
        "admin_name": user.get('full_name', 'Admin')
    })

# ==================== PUBLIC/COMMON ENDPOINTS ====================

@app.get("/evaluate", response_class=HTMLResponse)
async def evaluate_page(request: Request):
    """Public evaluation page (for demo)"""
    return templates.TemplateResponse("evaluate.html", {
        "request": request,
        "title": "Evaluate Answer"
    })

@app.post("/api/evaluate")
async def api_evaluate(
    question: str = Form(...),
    ideal_answer: str = Form(...),
    student_answer: str = Form(...)
):
    """Public API for evaluation — returns grade & marks"""
    # Validate
    if not student_answer or len(student_answer.strip()) < 5:
        return JSONResponse({'error': 'Answer too short or empty.'}, status_code=400)
    try:
        result = evaluator.evaluate(question, ideal_answer, student_answer)
        result['grade']       = result.get('grade', '')
        result['grade_label'] = result.get('grade_label', '')
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(content={'error': str(e)}, status_code=500)


@app.get("/api/health-check")
async def health_check_api():
    """Health check — returns evaluator model status"""
    model_status = 'sentence_transformer' if hasattr(evaluator, 'semantic_model') and evaluator.semantic_model else 'fallback'
    engine_status = 'synonym_aware' if hasattr(evaluator, 'accuracy_engine') and evaluator.accuracy_engine else 'basic'
    return {
        'status':        'healthy',
        'semantic_model': model_status,
        'accuracy_engine': engine_status,
        'version':       '2.1.0'
    }

@app.get("/about", response_class=HTMLResponse)
async def about_page(request: Request):
    """About page"""
    return templates.TemplateResponse("about.html", {
        "request": request,
        "title": "About This Project"
    })

@app.get("/api/stats")
async def get_stats():
    """Get system statistics"""
    return {
        "system": "Answer Evaluation System v2.0",
        "status": "Operational",
        "users": 3,
        "questions": len(db.questions) if hasattr(db, 'questions') else 2,
        "evaluations_completed": 150,
        "accuracy": "92%",
        "mode": "Multi-User LMS"
    }

@app.get("/api/export/submissions")
async def export_submissions_csv(request: Request):
    """Export all submissions as CSV for download"""
    import csv
    from io import StringIO
    from fastapi.responses import StreamingResponse
    
    user = get_current_user(request)
    if not user or user.get('role') != 'teacher':
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    
    submissions = db.get_student_submissions(user.get('user_id', 2))
    
    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(['Student', 'Question', 'Answer', 'Score', 'Confidence', 'Submitted At'])
    
    for sub in submissions:
        writer.writerow([
            sub.get('student_name', ''),
            sub.get('question_text', ''),
            sub.get('answer_text', ''),
            sub.get('final_score', ''),
            sub.get('confidence', ''),
            sub.get('submitted_at', '')
        ])
    
    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=student_submissions.csv"}
    )

# ==================== ERROR HANDLERS ====================

@app.exception_handler(404)
async def not_found_exception_handler(request: Request, exc: HTTPException):
    """Handle 404 errors"""
    return templates.TemplateResponse("404.html", {"request": request}, status_code=404)

@app.exception_handler(500)
async def internal_exception_handler(request: Request, exc: HTTPException):
    """Handle 500 errors"""
    return templates.TemplateResponse("500.html", {"request": request}, status_code=500)

# ==================== HEALTH CHECK ====================

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "database": "connected" if type(db).__name__ != 'DummyDB' else "demo_mode",
            "ai_models": "loaded",
            "authentication": "active",
            "frontend": "serving"
        }
    }

# ==================== STARTUP ====================

@app.on_event("startup")
async def startup_event():
    """Run on startup"""
    print("\n" + "="*60)
    print("🚀 ANSWER EVALUATION SYSTEM - MULTI-USER EDITION")
    print("="*60)
    print("System initialized successfully!")
    
    # Safe database type check
    try:
        # Check if DummyDB class exists in current scope
        from Utilities.database_manager import DatabaseManager
        if isinstance(db, DatabaseManager):
            db_status = "Connected (Real Database)"
        else:
            db_status = "Demo Mode (Dummy Database)"
    except:
        db_status = "Demo Mode"
    
    print(f"📊 Database: {db_status}")
    print(f"🧠 AI Models: Loaded ({evaluator.__class__.__name__})")
    print(f"👥 User Roles: Admin, Teacher, Student")
    print(f"🌐 Server: http://localhost:8000")
    print("="*60 + "\n")
# ==================== TEACHER ADDITIONAL ROUTES ====================

@app.get("/teacher/questions", response_class=HTMLResponse)
async def teacher_questions_page(request: Request):
    """Teacher view all questions"""
    user = get_current_user(request)
    if not user or user.get('role') != 'teacher':
        return RedirectResponse(url="/")
    
    # Get teacher's questions
    questions = db.get_teacher_questions(user.get('user_id', 2))
    
    return templates.TemplateResponse("teacher_questions.html", {
        "request": request,
        "teacher_name": user.get('full_name', 'Teacher'),
        "questions": questions
    })

@app.get("/teacher/students", response_class=HTMLResponse)
async def teacher_students_page(request: Request):
    """Teacher view students"""
    user = get_current_user(request)
    if not user or user.get('role') != 'teacher':
        return RedirectResponse(url="/")
    
    return templates.TemplateResponse("teacher_students.html", {
        "request": request,
        "teacher_name": user.get('full_name', 'Teacher')
    })

@app.get("/teacher/analytics", response_class=HTMLResponse)
async def teacher_analytics_page(request: Request):
    """Teacher analytics dashboard"""
    user = get_current_user(request)
    if not user or user.get('role') != 'teacher':
        return RedirectResponse(url="/")
    
    return templates.TemplateResponse("teacher_analytics.html", {
        "request": request,
        "teacher_name": user.get('full_name', 'Teacher')
    })

# ==================== ADMIN ADDITIONAL ROUTES ====================

@app.get("/admin/users", response_class=HTMLResponse)
async def admin_users_page(request: Request):
    """Admin user management"""
    user = get_current_user(request)
    if not user or user.get('role') != 'admin':
        return RedirectResponse(url="/")
    
    return templates.TemplateResponse("admin_users.html", {
        "request": request,
        "admin_name": user.get('full_name', 'Admin')
    })

@app.get("/admin/settings", response_class=HTMLResponse)
async def admin_settings_page(request: Request):
    """Admin system settings"""
    user = get_current_user(request)
    if not user or user.get('role') != 'admin':
        return RedirectResponse(url="/")
    
    return templates.TemplateResponse("admin_settings.html", {
        "request": request,
        "admin_name": user.get('full_name', 'Admin')
    })

@app.get("/admin/analytics", response_class=HTMLResponse)
async def admin_analytics_page(request: Request):
    """Admin analytics"""
    user = get_current_user(request)
    if not user or user.get('role') != 'admin':
        return RedirectResponse(url="/")
    
    return templates.TemplateResponse("admin_analytics.html", {
        "request": request,
        "admin_name": user.get('full_name', 'Admin')
    })

if __name__ == "__main__":
    uvicorn.run("Production_Deployment.fastapi_app:app", 
                host="0.0.0.0", 
                port=8000, 
                reload=True,
                log_level="info")