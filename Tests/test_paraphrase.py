"""
Test the cryptography paraphrase case - the exact user example.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from Advanced_Core.advanced_evaluator import AdvancedAnswerEvaluator

evaluator = AdvancedAnswerEvaluator()

cases = [
    {
        'label': 'Cryptography PARAPHRASE (exact user example)',
        'q': 'Define DSA',
        'ideal': 'the branch of computer science and mathematics dedicated to securing information and communication through the use of codes',
        'student': 'the science of hidden writing. It is the practice of scrambling a readable message so that only the person it was intended for can unscramble and read it.',
        'expect_min': 55,
    },
    {
        'label': 'Exact answer - should score very high',
        'q': 'What is cryptography?',
        'ideal': 'Cryptography is securing information using codes and mathematical techniques.',
        'student': 'Cryptography is securing information using codes and mathematical techniques.',
        'expect_min': 85,
    },
    {
        'label': 'Wrong topic - should stay LOW',
        'q': 'What is photosynthesis?',
        'ideal': 'Photosynthesis is the process plants use to convert sunlight into food.',
        'student': 'Cryptography is the use of codes to secure information from unauthorized access.',
        'expect_max': 35,
    },
    {
        'label': 'Different words, same meaning (general)',
        'q': 'Explain machine learning',
        'ideal': 'Machine learning is when computers learn patterns from data to make predictions.',
        'student': 'ML is a method where algorithms are trained on examples to forecast outcomes.',
        'expect_min': 55,
    },
]

print()
print("=" * 60)
print("CRYPTOGRAPHY PARAPHRASE TEST")
print("=" * 60)
all_ok = True
for c in cases:
    r = evaluator.evaluate(c['q'], c['ideal'], c['student'])
    score = r['final_score']
    grade = r['grade']
    sem   = r['layer_scores']['semantic']
    con   = r['layer_scores']['conceptual']

    passed = True
    if 'expect_min' in c:
        passed = score >= c['expect_min']
    else:
        passed = score <= c['expect_max']

    status = 'PASS' if passed else 'FAIL'
    print(f"\n[{status}] {c['label']}")
    print(f"  Score={score}/100  Grade={grade}")
    print(f"  Semantic={sem}  Conceptual={con}")
    print(f"  Feedback: {r['feedback'][:120]}")
    # Debug: accuracy engine detail
    acc = r.get('accuracy_details', {})
    if acc:
        print(f"  AccuracyScore={acc.get('accuracy_score',0):.3f}  ConceptPhrase={acc.get('concept_phrase_score',0):.3f}")
        print(f"  ConceptMatches={acc.get('concept_matches',[])}")
        print(f"  SynMatches={acc.get('synonym_matches',[])[:3]}")
    if not passed:
        all_ok = False

print()
print("=" * 60)
print("ALL OK" if all_ok else "SOME FAILED")
print("=" * 60)
