"""Quick integration test for the Neural Evaluator."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from Advanced_Core.advanced_evaluator import AdvancedAnswerEvaluator

evaluator = AdvancedAnswerEvaluator()

# Test 1: Good paraphrase
result = evaluator.evaluate(
    "What is photosynthesis?",
    "Photosynthesis is the process by which green plants use sunlight to synthesize food from carbon dioxide and water, producing oxygen.",
    "Plants make their own food using sunlight, CO2, and water. This biological process releases oxygen and stores energy."
)
print(f"TEST 1 (Paraphrase): Score={result['final_score']}, Grade={result['grade']}")

# Test 2: Wrong topic
result2 = evaluator.evaluate(
    "What is photosynthesis?",
    "Photosynthesis is the process by which green plants use sunlight to synthesize food from carbon dioxide and water.",
    "Animals eat food to get energy. They hunt prey in the jungle."
)
print(f"TEST 2 (Wrong Topic): Score={result2['final_score']}, Grade={result2['grade']}")

# Test 3: Exact answer
result3 = evaluator.evaluate(
    "What is machine learning?",
    "Machine learning is a subset of AI that enables systems to learn from data and improve from experience without being explicitly programmed.",
    "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed."
)
print(f"TEST 3 (Exact): Score={result3['final_score']}, Grade={result3['grade']}")

print("\n=== ALL TESTS COMPLETE ===")
