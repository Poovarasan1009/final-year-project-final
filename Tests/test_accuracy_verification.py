"""
Quick test script - ASCII output only (no emoji, Windows safe)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

def run_tests():
    from Advanced_Core.advanced_evaluator import AdvancedAnswerEvaluator, get_grade_band
    from Advanced_Core.accuracy_engine import AccuracyEngine, synonym_aware_overlap, words_are_synonymous

    evaluator = AdvancedAnswerEvaluator()
    acc_engine = AccuracyEngine()

    print("="*60)
    print("ANSWER EVALUATION SYSTEM - ACCURACY TEST SUITE")
    print("="*60)
    all_passed = True

    cases = [
        {
            "id": 1,
            "label": "Synonym: automobile = car",
            "q":   "What is an automobile?",
            "ideal": "An automobile is a motorized vehicle used for transportation on roads.",
            "student": "A car is a machine powered by an engine used to travel on roads.",
            "expect_min": 60,
        },
        {
            "id": 2,
            "label": "Correct answer - should score good (paraphrase)",
            "q":   "What are advantages of renewable energy?",
            "ideal": "Renewable energy reduces greenhouse gases, is sustainable, lowers costs, and creates jobs.",
            "student": "Renewable energy is eco-friendly, sustainable, creates employment, and reduces pollution.",
            "expect_min": 65,   # paraphrase, not verbatim — realistic bar
        },
        {
            "id": 3,
            "label": "Wrong answer - should score LOW",
            "q":   "What affects plant growth?",
            "ideal": "Plants need sunlight, water, and nutrients to grow well.",
            "student": "Animals grow fast when they eat meat and fish regularly.",
            "expect_max": 35,
        },
        {
            "id": 4,
            "label": "Empty answer - should score 0",
            "q":   "Explain gravity",
            "ideal": "Gravity is the force that attracts objects with mass toward each other.",
            "student": "  ",
            "expect_exact": 0,
        },
        {
            "id": 5,
            "label": "Synonym: rapid = fast",
            "q":   "Describe sorting algorithm performance",
            "ideal": "A fast sorting algorithm completes in O(n log n) time complexity.",
            "student": "A rapid sorting method finishes quickly with O(n log n) time.",
            "expect_min": 60,
        },
    ]

    for c in cases:
        result = evaluator.evaluate(c["q"], c["ideal"], c["student"])
        score  = result["final_score"]
        grade  = result["grade"]
        marks  = result.get("marks_obtained", 0)
        confid = result["confidence"]

        passed = True
        if "expect_min"   in c:
            passed = score >= c["expect_min"]
        elif "expect_max" in c:
            passed = score <= c["expect_max"]
        elif "expect_exact" in c:
            passed = score == c["expect_exact"]

        status = "PASS" if passed else "FAIL"
        print(f"\n[{status}] Test {c['id']}: {c['label']}")
        print(f"  Score: {score}/100  Grade: {grade}  Marks: {marks}/10  Confidence: {confid}%")

        # Print synonym matches if any
        acc = result.get("accuracy_details", {})
        syns = acc.get("synonym_matches", [])
        if syns:
            print(f"  Synonym matches detected: {[(s[0], s[1]) for s in syns[:3]]}")

        missed = result.get("details", {}).get("conceptual", {}).get("missed_concepts", [])
        if missed:
            print(f"  Missing concepts: {missed[:4]}")

        if not passed:
            all_passed = False

    # Grade band tests
    print("\n" + "="*60)
    print("GRADE BAND TESTS")
    print("="*60)
    bands = [(95, "A+"), (83, "A"), (72, "B"), (65, "C"), (52, "D"), (40, "E"), (20, "F")]
    for score_val, expected_grade in bands:
        gb = get_grade_band(score_val)
        ok = gb['grade'] == expected_grade
        print(f"  {'OK' if ok else 'FAIL'} Score {score_val} -> Grade {gb['grade']} ({gb['label']})")
        if not ok:
            all_passed = False

    # Synonym engine standalone test
    print("\n" + "="*60)
    print("SYNONYM ENGINE TESTS")
    print("="*60)
    syn_cases = [
        ("automobile", "car", True),
        ("rapid", "fast", True),
        ("happy", "planet", False),
        ("purchase", "buy", True),
    ]
    for w1, w2, expected in syn_cases:
        result = words_are_synonymous(w1, w2)
        ok = result == expected
        print(f"  {'OK' if ok else 'FAIL'} '{w1}' synonym of '{w2}'? Expected={expected}, Got={result}")
        if not ok:
            all_passed = False

    print("\n" + "="*60)
    print("ALL TESTS PASSED" if all_passed else "SOME TESTS FAILED - check above")
    print("="*60)
    return all_passed

if __name__ == "__main__":
    run_tests()
