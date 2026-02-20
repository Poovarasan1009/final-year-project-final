"""Test all teacher portal endpoints"""
import requests
import json
import time

BASE = "http://localhost:8000"
results = []
s = requests.Session()

# Login as teacher
r = s.post(f"{BASE}/api/login", data={"username": "teacher", "password": "teacher123", "role": "teacher"})
d = r.json()
results.append(f"Login: {'OK' if d.get('success') else 'FAIL - ' + d.get('error','')}")
if d.get('success'):
    s.cookies.set("access_token", d.get("token", ""))

# Test all teacher endpoints
endpoints = [
    "/teacher",
    "/teacher/questions",
    "/teacher/create-question",
    "/teacher/results",
    "/teacher/analytics",
    "/teacher/assignments",
    "/teacher/exams",
]

for ep in endpoints:
    try:
        r = s.get(f"{BASE}{ep}", timeout=10)
        has_err = 'internal server error' in r.text.lower() or 'traceback' in r.text.lower()
        status_icon = "PASS" if r.status_code == 200 and not has_err else "FAIL"
        line = f"{status_icon} | {ep} | Status={r.status_code} | Size={len(r.text)}"
        if has_err or r.status_code != 200:
            try:
                err_detail = json.loads(r.text).get('detail', r.text[:200])
            except:
                err_detail = r.text[:200]
            line += f" | Error: {err_detail}"
        results.append(line)
    except Exception as e:
        results.append(f"FAIL | {ep} | Exception: {str(e)[:100]}")
    time.sleep(0.3)

with open("Tests/teacher_test_results.txt", "w") as f:
    f.write("TEACHER PORTAL TEST RESULTS\n")
    f.write("=" * 70 + "\n")
    for line in results:
        f.write(line + "\n")
    f.write("=" * 70 + "\n")

print("Done - see Tests/teacher_test_results.txt")
