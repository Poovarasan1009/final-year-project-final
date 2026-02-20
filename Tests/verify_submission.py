import requests
import sys
import json

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

def verify_submission():
    print("Verifying Student Submission...")
    session = login("student", "student123", "student")
    if not session: return False
    
    # We need a valid question ID. Let's assume ID 1 exists.
    # Submit an answer
    print("Submitting answer for Question 1...")
    payload = {
        "question_id": 1,
        "answer_text": "Machine learning is a subset of artificial intelligence that focuses on building systems that learn from data."
    }
    
    response = session.post(f"{BASE_URL}/api/student/submit-answer", json=payload)
    
    if response.status_code != 200:
        print(f"❌ Submission failed with status {response.status_code}")
        print(response.text)
        return False
        
    result = response.json()
    print("Submission Result:", json.dumps(result, indent=2))
    
    # Check for keys
    required_keys = ['final_score', 'confidence', 'feedback', 'success']
    all_keys_present = True
    for key in required_keys:
        if key not in result:
            print(f"❌ Missing key: {key}")
            all_keys_present = False
        else:
            if key == 'success' and result[key] is not True:
                print(f"❌ Key present but False: {key}")
                all_keys_present = False
            else:
                print(f"✅ Key present: {key}")
            
    # Check score value
    score = result.get('final_score')
    if score is None:
        # Fallback check
        score = result.get('score')
        if score is not None:
            print(f"⚠️ 'score' key found (value: {score}), but 'final_score' was expected.")
        else:
            print("❌ No score found in result")
            return False
            
    print(f"✅ Evaluation Score: {score}")
    return all_keys_present

if __name__ == "__main__":
    try:
        if verify_submission():
            print("\n✅ Verification SUCCESSFUL")
            sys.exit(0)
        else:
            print("\n❌ Verification FAILED")
            sys.exit(1)
    except Exception as e:
        print(f"\n❌ Execution failed: {e}")
        sys.exit(1)
