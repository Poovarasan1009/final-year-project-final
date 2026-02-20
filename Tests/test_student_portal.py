"""Test all student portal endpoints - Output to file"""
import requests
import json
import time

BASE = "http://localhost:8000"
results = []

s = requests.Session()

# Login
r = s.post(f"{BASE}/api/login", data={"username": "student", "password": "student123", "role": "student"})
d = r.json()
results.append(f"Login: {'OK' if d.get('success') else 'FAIL - ' + d.get('error','')}")
if d.get('success'):
    s.cookies.set("access_token", d.get("token", ""))

endpoints = [
    "/student",
    "/student/questions", 
    "/student/performance",
    "/student/feedback",
    "/student/exams",
    "/student/question/1",
    "/student/assignment/1"
]

for ep in endpoints:
    try:
        r = s.get(f"{BASE}{ep}", timeout=10)
        has_err = 'internal server error' in r.text.lower() or 'traceback' in r.text.lower()
        status_icon = "PASS" if r.status_code == 200 and not has_err else "FAIL"
        line = f"{status_icon} | {ep} | Status={r.status_code} | Size={len(r.text)}"
        if has_err or r.status_code != 200:
            # Extract error info
            if 'detail' in r.text[:500]:
                try:
                    err_detail = json.loads(r.text).get('detail', r.text[:200])
                except:
                    err_detail = r.text[:200]
                line += f" | Error: {err_detail}"
        results.append(line)
    except Exception as e:
        results.append(f"FAIL | {ep} | Exception: {str(e)[:100]}")
    time.sleep(0.3)

# Write results to file
with open("Tests/test_results.txt", "w") as f:
    f.write("STUDENT PORTAL TEST RESULTS\n")
    f.write("=" * 60 + "\n")
    for line in results:
        f.write(line + "\n")
    f.write("=" * 60 + "\n")

print("Results written to Tests/test_results.txt")
