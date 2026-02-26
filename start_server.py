"""
start_server.py — Single command to launch the Answer Evaluation System
Run:  python start_server.py
"""
import os, sys, subprocess, time, webbrowser
from pathlib import Path

ROOT = Path(__file__).parent
os.chdir(ROOT)                          # Always run from project root
sys.path.insert(0, str(ROOT))

# ── ANSI colours ─────────────────────────────────────────────────────────────
G = "\033[92m"; Y = "\033[93m"; C = "\033[96m"; R = "\033[91m"; W = "\033[97m"
B = "\033[1m";  D = "\033[0m"

def pr(icon, msg, color=W): print(f"  {color}{B}{icon}{D}  {color}{msg}{D}")

# ── Banner ────────────────────────────────────────────────────────────────────
print(f"\n{C}{'═'*65}{D}")
print(f"{B}{C}   🎓  ANSWER EVALUATION SYSTEM  —  FINAL YEAR PROJECT{D}")
print(f"{C}   BE Computer Science & Engineering — Deep Learning AI{D}")
print(f"{C}{'═'*65}{D}\n")

# ── Dependency check ──────────────────────────────────────────────────────────
pr("🔍", "Checking required packages…", C)

missing = []
for pkg in ["fastapi","uvicorn","torch","sentence_transformers","nltk",
            "jinja2","colorama","numpy","sklearn"]:
    try: __import__(pkg)
    except ImportError: missing.append(pkg)

if missing:
    pr("⚠️ ", f"Installing: {', '.join(missing)}", Y)
    subprocess.run([sys.executable,"-m","pip","install","-q"]+missing, check=False)
else:
    pr("✅", "All packages present", G)

# ── NLTK data ─────────────────────────────────────────────────────────────────
pr("📦", "Ensuring NLTK data…", C)
import nltk
for corpus in ["punkt","stopwords","wordnet","averaged_perceptron_tagger"]:
    try: nltk.data.find(f"tokenizers/{corpus}")
    except LookupError:
        nltk.download(corpus, quiet=True)
pr("✅", "NLTK data ready", G)

# ── Static / template directories ────────────────────────────────────────────
os.makedirs("Frontend/static/css",  exist_ok=True)
os.makedirs("Frontend/static/js",   exist_ok=True)
os.makedirs("Frontend/templates",   exist_ok=True)
os.makedirs("Data",    exist_ok=True)
os.makedirs("Results", exist_ok=True)
pr("✅", "Directories confirmed", G)

# ── Quick sanity: import the app ──────────────────────────────────────────────
pr("🔧", "Importing FastAPI application…", C)
try:
    from Production_Deployment.fastapi_app import app   # noqa: F401
    pr("✅", "FastAPI app loaded successfully", G)
except Exception as exc:
    pr("❌", f"App import failed: {exc}", R)
    import traceback; traceback.print_exc()
    sys.exit(1)

# ── Launch ────────────────────────────────────────────────────────────────────
HOST, PORT = "127.0.0.1", 8000
print(f"\n{C}{'─'*65}{D}")
pr("🚀", f"Starting server on  http://{HOST}:{PORT}", G)
pr("👤", "Login credentials:", C)
print(f"""
       {W}┌─────────────┬──────────┬──────────────┐
       │  Role       │ Username │  Password    │
       ├─────────────┼──────────┼──────────────┤
       │  👨‍🏫 Teacher │ teacher  │  teacher123  │
       │  👨‍🎓 Student │ student  │  student123  │
       │  🔧 Admin   │ admin    │  admin123    │
       └─────────────┴──────────┴──────────────┘{D}
""")
pr("ℹ️ ", "Press  Ctrl+C  to stop the server", Y)
print(f"{C}{'─'*65}{D}\n")

# Open browser after a short delay (non-blocking)
def _open():
    time.sleep(3)
    webbrowser.open(f"http://{HOST}:{PORT}")

import threading
threading.Thread(target=_open, daemon=True).start()

# ── uvicorn ───────────────────────────────────────────────────────────────────
import uvicorn
uvicorn.run(
    "Production_Deployment.fastapi_app:app",
    host=HOST,
    port=PORT,
    reload=False,           # reload=True breaks on Windows with some setups
    log_level="info",
)
