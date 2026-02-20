#!/usr/bin/env python
"""
MAIN FILE - Run this to start everything
TOP 1% ANSWER EVALUATION SYSTEM
"""
import os
import sys
import json
from pathlib import Path
import torch
from colorama import init, Fore, Style
import pyfiglet

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Initialize colorama for colored output
init(autoreset=True)

def print_banner():
    """Display awesome banner"""
    banner = pyfiglet.figlet_format("ANSWER  EVALUATOR", font="slant")
    print(Fore.CYAN + banner)
    print(Fore.YELLOW + "=" * 70)
    print(Fore.GREEN + "TOP 1% FINAL YEAR PROJECT - BE CSE")
    print(Fore.GREEN + "Intelligent Descriptive Answer Evaluation System")
    print(Fore.YELLOW + "=" * 70)
    print()

def check_environment():
    """Check if all dependencies are installed"""
    print(Fore.BLUE + "[1/4] Checking environment...")
    
    checks = []
    
    # Check Python version
    py_version = sys.version_info
    checks.append(("Python 3.8+", f"{py_version.major}.{py_version.minor}", py_version.major >= 3 and py_version.minor >= 8))
    
    # Check PyTorch
    try:
        import torch
        checks.append(("PyTorch", torch.__version__, True))
    except:
        checks.append(("PyTorch", "Not found", False))
    
    # Check transformers
    try:
        import transformers
        checks.append(("Transformers", transformers.__version__, True))
    except:
        checks.append(("Transformers", "Not found", False))
    
    # Display checks
    print(f"{'Library':<20} {'Version':<15} {'Status':<10}")
    print("-" * 50)
    for name, version, status in checks:
        status_text = Fore.GREEN + "✓ OK" if status else Fore.RED + "✗ MISSING"
        print(f"{name:<20} {version:<15} {status_text}")
    
    # Check CUDA
    if torch.cuda.is_available():
        print(Fore.GREEN + f"\n✓ GPU available: {torch.cuda.get_device_name(0)}")
    else:
        print(Fore.YELLOW + "\n⚠ Using CPU (GPU not available)")
    
    return all(check[2] for check in checks)

def initialize_system():
    """Initialize all system components"""
    print(Fore.BLUE + "\n[2/4] Initializing system components...")
    
    # Create necessary directories
    directories = ['Data', 'Results', 'Results/plots', 'Results/tables', 'Frontend/static']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    print(Fore.GREEN + "✓ System directories created")
    
    # Initialize database
    try:
        from Utilities.database_manager import DatabaseManager
        db = DatabaseManager()
        db.initialize()
        print(Fore.GREEN + "✓ Database initialized")
    except Exception as e:
        print(Fore.YELLOW + f"⚠ Database init warning: {e}")
    
    # Load sample dataset
    try:
        from Real_Dataset.dataset_loader import load_sample_dataset
        dataset = load_sample_dataset()
        print(Fore.GREEN + f"✓ Sample dataset loaded ({len(dataset)} entries)")
    except:
        print(Fore.YELLOW + "⚠ Could not load sample dataset")
    
    return True

def run_demo():
    """Run a demo evaluation"""
    print(Fore.BLUE + "\n[3/4] Running demo evaluation...")
    
    try:
        from Advanced_Core.advanced_evaluator import AdvancedAnswerEvaluator
        
        # Initialize evaluator
        evaluator = AdvancedAnswerEvaluator()
        
        # Demo question
        question = "What is machine learning?"
        ideal_answer = "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. It focuses on developing algorithms that can access data and use it to learn for themselves."
        student_answer = "Machine learning is when computers learn from data to make decisions without being directly programmed."
        
        print(Fore.CYAN + "\n" + "=" * 60)
        print(Fore.WHITE + "DEMO EVALUATION")
        print(Fore.CYAN + "=" * 60)
        print(Fore.YELLOW + f"Question: {question}")
        print(Fore.GREEN + f"Ideal Answer: {ideal_answer[:100]}...")
        print(Fore.BLUE + f"Student Answer: {student_answer}")
        print(Fore.CYAN + "=" * 60)
        
        # Run evaluation
        result = evaluator.evaluate(question, ideal_answer, student_answer)
        
        # Display results
        print(Fore.MAGENTA + "\n📊 EVALUATION RESULTS:")
        print(Fore.WHITE + f"  Final Score: {Fore.GREEN}{result['final_score']}/100")
        print(Fore.WHITE + f"  Confidence: {result['confidence']}%")
        
        print(Fore.MAGENTA + "\n📈 LAYER SCORES:")
        for layer, score in result['layer_scores'].items():
            color = Fore.GREEN if score > 70 else Fore.YELLOW if score > 50 else Fore.RED
            print(f"  {layer.title():<15} {color}{score}/100")
        
        print(Fore.MAGENTA + "\n💡 FEEDBACK:")
        print(Fore.WHITE + f"  {result['feedback']}")
        
        print(Fore.CYAN + "\n" + "=" * 60)
        
        return True
        
    except Exception as e:
        print(Fore.RED + f"✗ Demo failed: {e}")
        return False

def show_menu():
    """Show main menu"""
    print(Fore.BLUE + "\n[4/4] MAIN MENU")
    print(Fore.YELLOW + "\nWhat would you like to do?")
    print(Fore.CYAN + "1. Start Web Interface (Recommended)")
    print(Fore.CYAN + "2. Run Batch Evaluation")
    print(Fore.CYAN + "3. View Documentation")
    print(Fore.CYAN + "4. Run Full Test Suite")
    print(Fore.CYAN + "5. Exit")
    
    choice = input(Fore.GREEN + "\nEnter choice (1-5): ").strip()
    
    if choice == "1":
        start_web_interface()
    elif choice == "2":
        run_batch_evaluation()
    elif choice == "3":
        show_documentation()
    elif choice == "4":
        run_tests()
    else:
        print(Fore.YELLOW + "\nExiting... Goodbye!")
        sys.exit(0)

def start_web_interface():
    """Start the FastAPI web interface"""
    print(Fore.GREEN + "\n🚀 Starting web interface...")
    print(Fore.YELLOW + "Open your browser and go to: " + Fore.CYAN + "http://localhost:8000")
    print(Fore.YELLOW + "Press Ctrl+C to stop the server")
    
    try:
        import uvicorn
        uvicorn.run("Production_Deployment.fastapi_app:app", 
                   host="0.0.0.0", port=8000, reload=True)
    except KeyboardInterrupt:
        print(Fore.YELLOW + "\nServer stopped.")
    except Exception as e:
        print(Fore.RED + f"Error starting server: {e}")

def run_batch_evaluation():
    """Run batch evaluation on sample dataset"""
    print(Fore.GREEN + "\nRunning batch evaluation...")
    
    try:
        from Comprehensive_Evaluation.batch_evaluator import BatchEvaluator
        evaluator = BatchEvaluator()
        results = evaluator.run()
        
        print(Fore.GREEN + f"\n✓ Evaluation complete!")
        print(Fore.CYAN + f"Processed {results['total']} answers")
        print(Fore.CYAN + f"Average score: {results['average_score']:.2f}/100")
        
        # Save results
        import pandas as pd
        df = pd.DataFrame(results['details'])
        df.to_csv('Results/batch_results.csv', index=False)
        print(Fore.GREEN + f"Results saved to Results/batch_results.csv")
        
    except Exception as e:
        print(Fore.RED + f"✗ Batch evaluation failed: {e}")

def show_documentation():
    """Show project documentation"""
    print(Fore.GREEN + "\n📚 PROJECT DOCUMENTATION")
    print(Fore.YELLOW + "\nKey Features:")
    print("1. 4-Layer Evaluation (Conceptual, Semantic, Structural, Completeness)")
    print("2. Domain-Adaptive AI Model")
    print("3. Real-time Feedback Generation")
    print("4. Statistical Validation")
    print("5. Web Interface & API")
    
    print(Fore.YELLOW + "\nProject Structure:")
    print("Advanced_Core/ - Main AI algorithms")
    print("Real_Dataset/ - Sample datasets")
    print("Production_Deployment/ - Web app & API")
    print("Comprehensive_Evaluation/ - Testing & validation")
    print("Thesis_Presentation/ - Reports & documentation")
    
    print(Fore.YELLOW + "\nTo get started:")
    print("1. Run: python main.py")
    print("2. Choose option 1 for web interface")
    print("3. Upload answers or type them in")
    print("4. View detailed evaluation results")
    
    input(Fore.GREEN + "\nPress Enter to continue...")

def run_tests():
    """Run test suite"""
    print(Fore.GREEN + "\n🧪 Running test suite...")
    
    try:
        import subprocess
        result = subprocess.run([sys.executable, "-m", "pytest", "Tests/", "-v"], 
                              capture_output=True, text=True)
        print(result.stdout)
        
        if result.returncode == 0:
            print(Fore.GREEN + "✓ All tests passed!")
        else:
            print(Fore.YELLOW + "⚠ Some tests failed")
            
    except Exception as e:
        print(Fore.RED + f"✗ Test error: {e}")

def main():
    """Main function"""
    try:
        print_banner()
        
        # Check environment
        if not check_environment():
            print(Fore.RED + "\n❌ Missing dependencies. Please install required packages.")
            print(Fore.YELLOW + "Run: pip install -r requirements.txt")
            return
        
        # Initialize system
        if not initialize_system():
            print(Fore.RED + "\n❌ System initialization failed.")
            return
        
        # Run demo
        run_demo()
        
        # Show menu
        show_menu()
        
    except KeyboardInterrupt:
        print(Fore.YELLOW + "\n\nInterrupted by user. Exiting...")
    except Exception as e:
        print(Fore.RED + f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()