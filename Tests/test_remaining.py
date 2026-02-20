"""Test teacher students and AI evaluator pages"""
import requests

BASE = "http://localhost:8000"
s = requests.Session()

# Login as teacher
r = s.post(f"{BASE}/api/login", data={"username": "teacher", "password": "teacher123", "role": "teacher"})
d = r.json()
print(f"Login: {'OK' if d.get('success') else 'FAIL'}")
if d.get("success"):
    s.cookies.set("access_token", d.get("token", ""))

# Test each endpoint
for ep in ["/teacher/students", "/evaluate"]:
    try:
        r = s.get(f"{BASE}{ep}", timeout=10)
        has_err = "internal server error" in r.text.lower() or "traceback" in r.text.lower()
        print(f"{ep}: status={r.status_code} size={len(r.text)} err={has_err}")
        if r.status_code != 200 or has_err:
            print(f"  DETAIL: {r.text[:300]}")
    except Exception as e:
        print(f"{ep}: EXCEPTION - {e}")
