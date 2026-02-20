import requests
import sys

BASE_URL = "http://localhost:8000"

def login(username, password, role):
    session = requests.Session()
    response = session.post(f"{BASE_URL}/api/login", data={
        "username": username,
        "password": password,
        "role": role
    })
    if response.status_code != 200 or not response.json().get('success'):
        print(f"Failed to login as {username}")
        return None
    return session

def verify_teacher_dashboard():
    print("Verifying Teacher Dashboard...")
    session = login("teacher", "teacher123", "teacher")
    if not session: return False
    
    response = session.get(f"{BASE_URL}/teacher")
    content = response.text
    
    checks = [
        ("Top Students Widget", "Top Students"),
        ("Topic Chart", 'id="topicChart"'),
        ("Analytics Data", "Active Students") # Matches <div class="stat-label">Active Students</div>
    ]
    
    all_passed = True
    for name, check in checks:
        if check in content:
            print(f"✅ {name} found")
        else:
            print(f"❌ {name} NOT found")
            all_passed = False
    return all_passed

def verify_student_question():
    print("\nVerifying Student Question Page...")
    session = login("student", "student123", "student")
    if not session: return False
    
    # We need a valid question ID. Let's assume ID 1 exists (created by default)
    response = session.get(f"{BASE_URL}/student/question/1")
    content = response.text
    
    checks = [
        ("Voice Input Button", 'id="voiceBtn"'),
        ("Radar Chart Canvas", 'id="layerChart"'),
        ("Gamification Badge", "Top 1% Candidate")
    ]
    
    all_passed = True
    for name, check in checks:
        if check in content:
            print(f"✅ {name} found")
        else:
            print(f"❌ {name} NOT found")
            all_passed = False
    return all_passed

if __name__ == "__main__":
    try:
        t_pass = verify_teacher_dashboard()
        s_pass = verify_student_question()
        
        if t_pass and s_pass:
            print("\n✅ All enhancements verified successfully!")
            sys.exit(0)
        else:
            print("\n❌ Some verifications failed.")
            sys.exit(1)
    except Exception as e:
        print(f"\n❌ Execution failed: {e}")
        sys.exit(1)
