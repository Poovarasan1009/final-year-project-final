"""Get error details from failing endpoints"""
import requests
import json

BASE = "http://localhost:8000"
s = requests.Session()

r = s.post(f"{BASE}/api/login", data={"username": "student", "password": "student123", "role": "student"})
d = r.json()
s.cookies.set("access_token", d.get("token", ""))

results = []
for ep in ["/student/performance", "/student/feedback", "/student/exams"]:
    r = s.get(f"{BASE}{ep}", timeout=10)
    results.append(f"{ep}: Status={r.status_code}")
    results.append(f"  Body: {r.text[:300]}")
    results.append("")

with open("Tests/error_details.txt", "w") as f:
    for line in results:
        f.write(line + "\n")

print("Done - see Tests/error_details.txt")
