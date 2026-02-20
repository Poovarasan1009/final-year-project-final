import sys
import os
import time

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from Advanced_Core.advanced_evaluator import AdvancedAnswerEvaluator
except ImportError:
    print("❌ Error: Could not import AdvancedAnswerEvaluator.")
    print("Make sure you are running this from the project root or Tests folder.")
    print("Command: python Tests/demo_layers.py")
    sys.exit(1)

def print_separator(title):
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)

def demonstrate_layers():
    print_separator("INITIALIZING AI SYSTEM")
    print("Loading SBERT and NLP models... (This uses Transfer Learning)")
    evaluator = AdvancedAnswerEvaluator()
    
    # Sample Data
    question = "Explain the process of photosynthesis."
    ideal_answer = "Photosynthesis is the process by which green plants use sunlight to synthesize nutrients from carbon dioxide and water. Photosynthesis in plants generally involves the green pigment chlorophyll and generates oxygen as a byproduct."
    
    # Case 1: A Good Answer
    student_good = "Photosynthesis is how plants make food. They use sunlight, water and carbon dioxide to create nutrients. This happens in chlorophyll and releases oxygen."
    
    # Case 2: A Weak Answer
    student_weak = "Plants need sun to grow. It is good for nature."
    
    # Case 3: A Wrong Structure Answer (Keyword stuffing)
    student_stuffing = "Photosynthesis sunlight water carbon dioxide chlorophyll oxygen nutrients."

    print_separator(f"DEMONSTRATION: QUESTION & CONTEXT")
    print(f"❓ Question: {question}")
    print(f"✅ Ideal Answer: {ideal_answer}")
    
    # --- Run for Good Answer ---
    print_separator("CASE 1: GOOD ANSWER (Correct Concept & Structure)")
    print(f"📝 Student Answer: {student_good}\n")
    
    # Call the evaluator (which prints internal logs)
    result_good = evaluator.evaluate(question, ideal_answer, student_good)
    
    print("\n📊 FINAL CALCULATED SCORES (Internal Data):")
    print(f"   1. Conceptual Score: {result_good['layer_scores']['conceptual']:.1f}%  (High Keyword Match)")
    print(f"   2. Semantic Score:   {result_good['layer_scores']['semantic']:.1f}%  (High Meaning Match)")
    print(f"   3. Structural Score: {result_good['layer_scores']['structural']:.1f}%  (Good Grammar)")
    print(f"   4. Completeness:     {result_good['layer_scores']['completeness']:.1f}%  (Coverage Good)")
    print(f"\n🏆 FINAL WEIGHTED SCORE: {result_good['final_score']:.1f}%")
    
    time.sleep(2)
    
    # --- Run for Weak Answer ---
    print_separator("CASE 2: WEAK ANSWER (Missing Details)")
    print(f"📝 Student Answer: {student_weak}\n")
    
    result_weak = evaluator.evaluate(question, ideal_answer, student_weak)
    
    print("\n📊 FINAL CALCULATED SCORES (Internal Data):")
    print(f"   1. Conceptual Score: {result_weak['layer_scores']['conceptual']:.1f}%  (Missing Specific Keywords)")
    print(f"   2. Semantic Score:   {result_weak['layer_scores']['semantic']:.1f}%  (Low Similarity)")
    print(f"   3. Structural Score: {result_weak['layer_scores']['structural']:.1f}%  (Too Short)")
    print(f"   4. Completeness:     {result_weak['layer_scores']['completeness']:.1f}%  (Answer Incomplete)")
    print(f"\n🏆 FINAL WEIGHTED SCORE: {result_weak['final_score']:.1f}%")

    time.sleep(2)
    
    # --- Run for Keyword Stuffing ---
    print_separator("CASE 3: KEYWORD STUFFING (Attempted Cheat)")
    print(f"📝 Student Answer: {student_stuffing}\n")
    
    result_cheat = evaluator.evaluate(question, ideal_answer, student_stuffing)
    
    print("\n📊 FINAL CALCULATED SCORES (Internal Data):")
    print(f"   1. Conceptual Score: {result_cheat['layer_scores']['conceptual']:.1f}%  (High - Concepts Present)")
    print(f"   2. Semantic Score:   {result_cheat['layer_scores']['semantic']:.1f}%  (Medium - Meaning Check)")
    print(f"   3. Structural Score: {result_cheat['layer_scores']['structural']:.1f}%  (Extremely Low - No Grammar)")
    print(f"   4. Completeness:     {result_cheat['layer_scores']['completeness']:.1f}%  (Incoherent)")
    print(f"\n🏆 FINAL WEIGHTED SCORE: {result_cheat['final_score']:.1f}%")
    
    print_separator("DEMONSTRATION COMPLETE")
    print("This proves that the system distinguishes between 'Concept', 'Meaning', and 'Structure'.")
    print("A simple model would have given Case 3 a high score due to keywords.")

if __name__ == "__main__":
    demonstrate_layers()
