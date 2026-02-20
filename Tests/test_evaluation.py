"""
Test suite for Answer Evaluation System
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from Advanced_Core.advanced_evaluator import AdvancedAnswerEvaluator

def test_basic_evaluation():
    """Test basic answer evaluation"""
    print("🧪 Testing Basic Evaluation...")
    
    evaluator = AdvancedAnswerEvaluator()
    
    test_cases = [
        {
            "question": "What is photosynthesis?",
            "ideal": "Photosynthesis is the process by which plants convert sunlight into chemical energy.",
            "student": "Plants use sunlight to make food through photosynthesis.",
            "expected_min_score": 70
        },
        {
            "question": "Define machine learning",
            "ideal": "Machine learning is a subset of AI that enables systems to learn from experience.",
            "student": "Machine learning is when computers learn from data.",
            "expected_min_score": 60
        },
        {
            "question": "What affects plant growth?",
            "ideal": "Plants grow more in sunlight",
            "student": "Animals grow more in sunlight",  # Wrong answer
            "expected_max_score": 40
        }
    ]
    
    all_passed = True
    
    for i, test in enumerate(test_cases, 1):
        result = evaluator.evaluate(test["question"], test["ideal"], test["student"])
        score = result["final_score"]
        
        if "expected_min_score" in test:
            if score >= test["expected_min_score"]:
                print(f"  ✅ Test {i} PASSED: Score {score} >= {test['expected_min_score']}")
            else:
                print(f"  ❌ Test {i} FAILED: Score {score} < {test['expected_min_score']}")
                all_passed = False
        elif "expected_max_score" in test:
            if score <= test["expected_max_score"]:
                print(f"  ✅ Test {i} PASSED: Score {score} <= {test['expected_max_score']}")
            else:
                print(f"  ❌ Test {i} FAILED: Score {score} > {test['expected_max_score']}")
                all_passed = False
    
    return all_passed

def test_4_layer_analysis():
    """Test that all 4 layers return scores"""
    print("\n🧪 Testing 4-Layer Analysis...")
    
    evaluator = AdvancedAnswerEvaluator()
    result = evaluator.evaluate(
        "What is AI?",
        "Artificial Intelligence is the simulation of human intelligence by machines.",
        "AI makes computers smart like humans."
    )
    
    layers = ["conceptual", "semantic", "structural", "completeness"]
    missing_layers = []
    
    for layer in layers:
        if layer not in result.get("layer_scores", {}):
            missing_layers.append(layer)
    
    if not missing_layers:
        print("  ✅ All 4 layers present in results")
        for layer, score in result["layer_scores"].items():
            print(f"    {layer}: {score}/100")
        return True
    else:
        print(f"  ❌ Missing layers: {missing_layers}")
        return False

def test_confidence_scoring():
    """Test confidence scoring"""
    print("\n🧪 Testing Confidence Scoring...")
    
    evaluator = AdvancedAnswerEvaluator()
    result = evaluator.evaluate(
        "Explain gravity",
        "Gravity is the force that attracts objects with mass toward each other.",
        "Gravity pulls things down."
    )
    
    confidence = result.get("confidence", 0)
    
    if 0 <= confidence <= 100:
        print(f"  ✅ Confidence score valid: {confidence}%")
        return True
    else:
        print(f"  ❌ Invalid confidence score: {confidence}")
        return False

def run_all_tests():
    """Run all tests"""
    print("="*60)
    print("🚀 RUNNING COMPREHENSIVE TEST SUITE")
    print("="*60)
    
    tests = [
        test_basic_evaluation,
        test_4_layer_analysis,
        test_confidence_scoring
    ]
    
    results = []
    for test in tests:
        try:
            results.append(test())
        except Exception as e:
            print(f"  ❌ Test failed with error: {e}")
            results.append(False)
    
    passed = sum(results)
    total = len(results)
    
    print("\n" + "="*60)
    print(f"📊 TEST RESULTS: {passed}/{total} tests passed")
    print("="*60)
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! System is working correctly.")
        return True
    else:
        print("⚠ Some tests failed. Check the implementation.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)