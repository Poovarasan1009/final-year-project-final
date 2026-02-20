import sys
sys.path.insert(0, '.')
from Advanced_Core.advanced_evaluator import AdvancedAnswerEvaluator

e = AdvancedAnswerEvaluator()

question = "What is DSA?"
ideal = "DSA is the study of resource optimization in computing. Data Structures provide a specialized format for organizing processing and retrieving data so that we can access it efficiently. Algorithms are the logic applied to that data to transform an input into an output."
student = "DSA stands for Data Structures and Algorithms. Data Structures are ways to store and organize data like Arrays Linked Lists or Stacks. Algorithms are the steps or rules we follow to solve a problem like sorting a list or searching for a value. We learn DSA mainly to pass coding interviews and write code that runs faster."

r = e.evaluate(question, ideal, student)

print("\n" + "="*50)
print("RESULTS:")
print("="*50)
print(f"L1 Conceptual:   {r['layer_scores']['conceptual']}%")
print(f"L2 Semantic:     {r['layer_scores']['semantic']}%")
print(f"L3 Structural:   {r['layer_scores']['structural']}%")
print(f"L4 Completeness: {r['layer_scores']['completeness']}%")
print(f"Question Type:   {r['question_type']}")
print(f"Weights:         {r['weights']}")
print(f"FINAL SCORE:     {r['final_score']}%")
print(f"Confidence:      {r['confidence']}%")
print(f"Feedback:        {r['feedback']}")
print("="*50)
