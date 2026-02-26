"""
TOP 1% AI MODEL - Advanced Answer Evaluator (Production Grade)
4-Layer Evaluation + Synonym-Aware Accuracy Engine
"""
# Torch is optional — if its native DLLs are broken we fall back to sklearn
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_OK = True
except Exception:
    TORCH_OK = False
import numpy as np
import re
from typing import Dict, List, Tuple
from collections import Counter
import json
import warnings
warnings.filterwarnings('ignore')


# ── NLTK Stopwords (comprehensive) ─────────────────────────────────────────────
STOPWORDS = {
    'a', 'an', 'the', 'and', 'or', 'but', 'is', 'are', 'was', 'were', 'be',
    'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
    'could', 'should', 'may', 'might', 'shall', 'can', 'to', 'of', 'in', 'for',
    'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during', 'before',
    'after', 'above', 'below', 'this', 'that', 'these', 'those', 'it', 'its',
    'we', 'they', 'he', 'she', 'you', 'i', 'me', 'him', 'her', 'us', 'them',
    'my', 'our', 'your', 'his', 'their', 'which', 'who', 'what', 'when', 'where',
    'how', 'if', 'than', 'so', 'such', 'not', 'no', 'also', 'just', 'very',
    'only', 'even', 'about', 'each', 'both', 'all', 'any', 'some', 'most',
    'more', 'other', 'then', 'than', 'now', 'here', 'there', 'up', 'down',
    'get', 'got', 'make', 'made', 'said', 'say', 'use', 'used', 'using'
}


def get_grade_band(score: float) -> Dict:
    """Convert percentage score to a grade band with label and description."""
    if score >= 90:
        return {'grade': 'A+', 'label': 'Outstanding',    'color': '#00c853'}
    elif score >= 80:
        return {'grade': 'A',  'label': 'Excellent',      'color': '#64dd17'}
    elif score >= 70:
        return {'grade': 'B',  'label': 'Good',           'color': '#aeea00'}
    elif score >= 60:
        return {'grade': 'C',  'label': 'Average',        'color': '#ffd600'}
    elif score >= 50:
        return {'grade': 'D',  'label': 'Below Average',  'color': '#ff9100'}
    elif score >= 35:
        return {'grade': 'E',  'label': 'Poor',           'color': '#ff3d00'}
    else:
        return {'grade': 'F',  'label': 'Fail',           'color': '#b71c1c'}


def marks_from_score(score: float, max_marks: int) -> float:
    """Convert percentage score to actual marks."""
    return round((score / 100.0) * max_marks, 1)


class AdvancedAnswerEvaluator:
    """
    PRODUCTION-GRADE 4-Layer Evaluation System:
      Layer 1 - Conceptual Understanding  : Key concept coverage (synonym-aware)
      Layer 2 - Semantic Similarity       : Sentence-BERT cosine similarity
      Layer 3 - Structural Coherence      : Sentence structure, connectors, grammar
      Layer 4 - Completeness Assessment   : Coverage of question requirements

    + AccuracyEngine integration for synonym/n-gram boost
    """

    def __init__(self, use_gpu: bool = False):
        if TORCH_OK:
            self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
            print(f"🚀 Using device: {self.device}")
        else:
            self.device = None
            print("[INFO] PyTorch unavailable — using sklearn/TF-IDF fallback mode")

        self.semantic_model = None
        self.transformer_model = None
        self.tokenizer = None

        # Load AI models
        self.load_models()

        # Load AccuracyEngine for synonym-aware scoring
        try:
            from Advanced_Core.accuracy_engine import AccuracyEngine
            self.accuracy_engine = AccuracyEngine(use_wordnet=True)
            print("[OK] Accuracy Engine (synonym-aware) loaded")
        except Exception as e:
            print(f"[WARN] Accuracy engine warn: {e}")
            self.accuracy_engine = None

        # Load Custom Deep Learning Neural Evaluator (requires working torch)
        self.neural_evaluator = None
        if TORCH_OK:
            try:
                from Advanced_Core.neural_evaluator import NeuralEvaluator
                self.neural_evaluator = NeuralEvaluator()
                if self.neural_evaluator.is_loaded():
                    print("[OK] Custom DL Neural Evaluator loaded")
                else:
                    print("[INFO] Neural evaluator not trained yet — using rule-based scoring")
                    self.neural_evaluator = None
            except Exception as e:
                print(f"[WARN] Neural evaluator warn: {e}")
                self.neural_evaluator = None
        else:
            print("[INFO] Neural evaluator skipped (PyTorch unavailable)")

        # Academic vocabulary (domain-specific boost terms)
        self.academic_terms = {
            'hypothesis', 'theory', 'experiment', 'analysis', 'conclusion',
            'methodology', 'results', 'discussion', 'variable', 'constant',
            'control', 'dependent', 'independent', 'quantitative', 'qualitative',
            'photosynthesis', 'mitosis', 'algorithm', 'database', 'neural',
            'network', 'optimization', 'complexity', 'efficiency', 'sustainable',
            'renewable', 'environmental', 'economic', 'social', 'political',
            'entropy', 'momentum', 'velocity', 'acceleration', 'equilibrium',
            'oxidation', 'reduction', 'derivative', 'integral', 'recursion',
            'inheritance', 'polymorphism', 'encapsulation'
        }

        # Connector words for structure analysis
        self.connectors = [
            'however', 'therefore', 'because', 'thus', 'consequently',
            'furthermore', 'moreover', 'in addition', 'firstly', 'secondly',
            'finally', 'in conclusion', 'for example', 'specifically',
            'on the other hand', 'similarly', 'conversely', 'additionally',
            'as a result', 'nevertheless', 'hence', 'whereas', 'although'
        ]

        # Question type classifiers
        self.question_patterns = {
            'definition':    ['define', 'what is', 'meaning of', 'explain the term', 'what are'],
            'explanation':   ['explain', 'describe', 'discuss', 'elaborate', 'how does'],
            'comparison':    ['compare', 'contrast', 'difference between', 'similarities', 'distinguish'],
            'advantages':    ['advantages', 'benefits', 'pros', 'strengths', 'merits'],
            'disadvantages': ['disadvantages', 'drawbacks', 'cons', 'limitations', 'demerits'],
            'process':       ['how', 'process', 'steps', 'procedure', 'mechanism', 'stages'],
            'analysis':      ['analyze', 'analyse', 'evaluate', 'assess', 'critically']
        }

    # ───────────────────────── Model Loading ─────────────────────────────────

    def load_models(self):
        """Load AI models for semantic evaluation."""
        print("[INFO] Loading AI models...")

        try:
            from sentence_transformers import SentenceTransformer
            self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
            if self.device and TORCH_OK:
                self.semantic_model = self.semantic_model.to(self.device)
            print("[OK] Semantic model loaded (all-MiniLM-L6-v2)")
        except Exception as e:
            print(f"[WARN] Semantic model not available: {e}")
            self.semantic_model = None

        try:
            from transformers import AutoTokenizer, AutoModel
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            self.transformer_model = AutoModel.from_pretrained("bert-base-uncased").to(self.device)
            print("[OK] Contextual BERT model loaded")
        except Exception as e:
            print(f"[WARN] Transformer model not available: {e}")
            self.transformer_model = None

    # ───────────────────────── Main Evaluate ─────────────────────────────────

    def evaluate(self, question: str, ideal_answer: str, student_answer: str,
                 subject: str = None, max_marks: int = 10) -> Dict:
        """
        Main evaluation function — returns comprehensive results.
        Handles empty/garbage answers gracefully.
        """
        # ── Guard: too short answers ──
        if not student_answer or len(student_answer.strip()) < 5:
            grade = get_grade_band(0)
            return {
                'final_score': 0.0,
                'confidence': 0.0,
                'grade': grade['grade'],
                'grade_label': grade['label'],
                'marks_obtained': 0.0,
                'max_marks': max_marks,
                'layer_scores': {'conceptual': 0, 'semantic': 0, 'structural': 0, 'completeness': 0},
                'weights': [0.35, 0.35, 0.15, 0.15],
                'question_type': 'general',
                'feedback': '❌ Answer is too short or empty. Please write a proper answer.',
                'feedback_detailed': {},
                'accuracy_details': {},
                'details': {}
            }

        # Keep original text for structure analysis; clean copy for concept work
        student_orig  = student_answer
        ideal_orig    = ideal_answer
        question_orig = question

        question_clean  = self.preprocess_text(question)
        ideal_clean     = self.preprocess_text(ideal_answer)
        student_clean   = self.preprocess_text(student_answer)

        print(f"\n[INFO] Evaluating answer...")

        # ── Layer 1: Conceptual Understanding ──
        print("   [L1] Concept coverage (synonym-aware)...")
        concept_score, concept_details = self.evaluate_concepts(
            question_clean, ideal_clean, student_clean,
            ideal_orig=ideal_orig, student_orig=student_orig
        )

        # ── Layer 2: Semantic Similarity ──
        print("   [L2] Semantic meaning...")
        semantic_score, semantic_details = self.evaluate_semantics(ideal_orig, student_orig)

        # ── Layer 3: Structural Coherence ──
        print("   [L3] Structure & grammar...")
        structure_score, structure_details = self.evaluate_structure(student_orig)

        # ── Layer 4: Completeness ──
        print("   [L4] Completeness...")
        completeness_score, completeness_details = self.evaluate_completeness(
            question_clean, student_clean, ideal_clean
        )

        # ── Accuracy Engine (synonym + n-gram + domain boost) ──
        accuracy_boost      = 0.0
        concept_phrase_score = 0.0
        accuracy_details    = {}
        if self.accuracy_engine:
            try:
                acc = self.accuracy_engine.score(question_orig, ideal_orig, student_orig, subject)
                accuracy_details     = acc
                accuracy_boost       = acc.get('accuracy_score', 0.0)
                concept_phrase_score = acc.get('concept_phrase_score', 0.0)

                # Larger boost per synonym match found
                if acc.get('synonym_matches'):
                    bonus = len(acc['synonym_matches']) * 0.08
                    concept_score = min(concept_score + bonus, 1.0)

                # ── KEY FIX: Concept-phrase override ──────────────────────────
                # If concept_phrase engine found high coverage of the domain
                # concepts, that is a STRONG signal the student gets it right.
                # Re-calculate accuracy_boost to reflect this:
                if concept_phrase_score >= 0.67:
                    # Majority of domain concepts found in student answer
                    # Weight: 60% concept_phrase, 40% raw accuracy
                    accuracy_boost = 0.60 * concept_phrase_score + 0.40 * accuracy_boost
                    print(f"   [OVERRIDE] Concept-phrase coverage={concept_phrase_score:.2f} — accuracy_boost raised to {accuracy_boost:.2f}")

            except Exception as e:
                print(f"   [WARN] Accuracy engine error: {e}")


        # ── SEMANTIC OVERRIDE gate ────────────────────────────────────────────
        # Only apply SBERT override if there is CORROBORATING evidence:
        # - concept_phrase_score > 0  (domain concepts matched in student answer)
        # - OR accuracy_boost >= 0.2  (synonym/n-gram are non-trivial)
        # - OR concept_score already >= 0.30 (student got some keywords right)
        # WITHOUT this gate, similar-vocabulary wrong-topic answers get boosted:
        #   e.g. "animals grow fast eating meat" vs "plants need sunlight to grow"
        #   (SBERT ≈ 0.60 due to shared word 'grow' — but should stay LOW)
        # ─────────────────────────────────────────────────────────────────────
        sbert_override_ok = (
            concept_phrase_score > 0.0      # domain phrases matched
            or accuracy_boost   >= 0.30     # meaningful keyword/synonym overlap
            or concept_score    >= 0.35     # student already matched some concepts
            or semantic_score   >= 0.75     # HIGH CONFIDENCE FALLBACK (e.g. ML case)
        )

        if sbert_override_ok:
            if semantic_score >= 0.60:
                concept_score = max(concept_score, 0.70)
                print("   [OVERRIDE] High semantic similarity — concept floor raised to 0.70")
            elif semantic_score >= 0.45:
                concept_score = max(concept_score, 0.55)
                print("   [OVERRIDE] Good semantic similarity — concept floor raised to 0.55")
            elif semantic_score >= 0.38:
                concept_score = max(concept_score, 0.40)
                print("   [OVERRIDE] Moderate semantic similarity — concept floor raised to 0.40")

        # Accuracy engine rescue: if strong synonym/concept match, raise concept floor
        if accuracy_boost >= 0.45 and concept_score < 0.55:
            concept_score = max(concept_score, accuracy_boost * 0.90)
            print(f"   [OVERRIDE] Accuracy engine boost={accuracy_boost:.2f} — concept floor raised")

        # ── Dynamic Weights ──
        question_type = self.classify_question(question_orig)
        weights = self.get_dynamic_weights(question_type)

        # ── Final Score (Rule-Based) ──
        raw_score = (
            weights[0] * concept_score +
            weights[1] * semantic_score +
            weights[2] * structure_score +
            weights[3] * completeness_score
        )

        # Blend with accuracy engine output (30% blend)
        if accuracy_boost > 0:
            raw_score = 0.70 * raw_score + 0.30 * accuracy_boost

        # Semantic floor: final score should never be below 75% of SBERT score.
        # GATED: only apply if we have corroborating evidence (sbert_override_ok).
        # This prevents correct paraphrase answers from scoring below their rating
        # while keeping wrong-topic answers from being unfairly rescued.
        if sbert_override_ok:
            semantic_floor = semantic_score * 0.75
            raw_score = max(raw_score, semantic_floor)

        rule_based_score = min(raw_score * 100, 100.0)

        # ── Neural Network Prediction (Deep Learning) ──
        neural_score = None
        if self.neural_evaluator and self.neural_evaluator.is_loaded():
            ideal_words = len(ideal_orig.split())
            student_words = len(student_orig.split())
            word_count_ratio = min(student_words / max(ideal_words, 1), 2.0)

            nn_features = {
                "concept_score": concept_score,
                "semantic_score": semantic_score,
                "structure_score": structure_score,
                "completeness_score": completeness_score,
                "word_count_ratio": word_count_ratio,
                "accuracy_boost": accuracy_boost,
                "concept_phrase_score": concept_phrase_score,
            }
            neural_score = self.neural_evaluator.predict(nn_features)
            print(f"   [DL] Neural model prediction: {neural_score:.2f}")

        # ── Hybrid Score: Blend Rule-Based + Neural ──
        if neural_score is not None:
            # 25% Neural Network + 75% Rule-Based for stability
            blended = 0.25 * neural_score + 0.75 * rule_based_score
            # Safety floor: neural model cannot drag score more than 15 pts below rule-based
            final_score = max(blended, rule_based_score - 15.0)
            print(f"   [HYBRID] Rule={rule_based_score:.2f}, Neural={neural_score:.2f}, Final={final_score:.2f}")
        else:
            final_score = rule_based_score

        final_score = max(0.0, min(final_score, 100.0))

        # ── Grade & Marks ──
        grade_info = get_grade_band(final_score)
        marks_obtained = marks_from_score(final_score, max_marks)

        # ── Confidence ──
        confidence = self.calculate_confidence(
            [concept_score, semantic_score, structure_score, completeness_score]
        )

        # ── Detailed Feedback ──
        feedback_simple, feedback_detailed = self.generate_feedback(
            concept_score, semantic_score, structure_score, completeness_score,
            concept_details, completeness_details, accuracy_details
        )

        print(f"   [RESULT] Final Score: {round(final_score,2)}/100  Grade: {grade_info['grade']}")
        print(f"   [RESULT] Confidence: {confidence}%")

        return {
            'final_score':      round(final_score, 2),
            'confidence':       confidence,
            'grade':            grade_info['grade'],
            'grade_label':      grade_info['label'],
            'marks_obtained':   marks_obtained,
            'max_marks':        max_marks,
            'layer_scores': {
                'conceptual':   round(concept_score * 100, 2),
                'semantic':     round(semantic_score * 100, 2),
                'structural':   round(structure_score * 100, 2),
                'completeness': round(completeness_score * 100, 2)
            },
            'weights':          [round(w, 3) for w in weights],
            'question_type':    question_type,
            'feedback':         feedback_simple,
            'feedback_detailed': feedback_detailed,
            'accuracy_details': accuracy_details,
            'details': {
                'conceptual':   concept_details,
                'semantic':     semantic_details,
                'structural':   structure_details,
                'completeness': completeness_details
            }
        }

    # ───────────────────────── Preprocessing ─────────────────────────────────

    def preprocess_text(self, text: str) -> str:
        """
        Clean text for KEYWORD extraction only.
        Keeps words, removes extra spaces. Does NOT remove sentence punctuation
        (that is preserved in original text for structure analysis).
        """
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)   # remove punctuation for keyword work
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    # ───────────────────── Layer 1: Conceptual ───────────────────────────────

    def evaluate_concepts(self, question: str, ideal: str, student: str,
                          ideal_orig: str = None, student_orig: str = None) -> Tuple[float, Dict]:
        """
        Layer 1: Conceptual understanding.
        Uses synonym-aware matching so 'automobile' == 'car', 'rapid' == 'fast'.
        """
        # Extract key concepts
        ideal_concepts   = self.extract_concepts(ideal, top_n=15)
        student_concepts = self.extract_concepts(student, top_n=20)
        question_concepts = self.extract_concepts(question, top_n=5)

        required_concepts = set(ideal_concepts + question_concepts)
        student_concepts_set = set(student_concepts)

        if not required_concepts:
            return 0.5, {'message': 'No key concepts identified'}

        # ── Exact match ──
        exact_matched = required_concepts.intersection(student_concepts_set)

        # ── Synonym-aware match on remaining ──
        synonym_matched = set()
        synonym_pairs   = []
        remaining = required_concepts - exact_matched
        for req in remaining:
            for sc in student_concepts_set:
                if self._synonymous(req, sc):
                    synonym_matched.add(req)
                    synonym_pairs.append((req, sc))
                    break

        all_matched = exact_matched | synonym_matched
        coverage = len(all_matched) / len(required_concepts)

        # ── Weighted by importance ──
        concept_importance = self.calculate_concept_importance(ideal, list(required_concepts))
        weighted_score = sum(
            concept_importance.get(c, 0.1)
            for c in all_matched
        )
        total_importance = sum(concept_importance.values())
        final_score = (weighted_score / total_importance) if total_importance > 0 else coverage

        # ── Academic bonus ──
        academic_hits = sum(1 for t in self.academic_terms if t in student)
        academic_bonus = min(academic_hits * 0.03, 0.10)
        final_score = min(final_score + academic_bonus, 1.0)

        details = {
            'required_concepts':  list(required_concepts),
            'student_concepts':   list(student_concepts_set),
            'matched_exact':      list(exact_matched),
            'matched_synonym':    list(synonym_matched),
            'synonym_pairs':      synonym_pairs,
            'missed_concepts':    list(remaining - synonym_matched),
            'coverage_percentage': round(coverage * 100, 1),
            'academic_bonus':     academic_bonus
        }

        return min(final_score, 1.0), details

    def _synonymous(self, word1: str, word2: str) -> bool:
        """Check if two words are synonymous via WordNet or stem."""
        if word1 == word2:
            return True
        # Stem check (first 5 chars)
        if len(word1) >= 5 and len(word2) >= 5 and word1[:5] == word2[:5]:
            return True
        try:
            from Advanced_Core.accuracy_engine import words_are_synonymous
            return words_are_synonymous(word1, word2)
        except Exception:
            return False

    # ───────────────────── Layer 2: Semantic ─────────────────────────────────

    def evaluate_semantics(self, ideal: str, student: str) -> Tuple[float, Dict]:
        """
        Layer 2: Semantic similarity via Sentence-BERT.

        Uses TWO scores and takes the higher one:
          1. Full-sentence SBERT cosine similarity (whole paragraph vs whole paragraph)
          2. Phrase-level SBERT similarity (best-matching key phrase from ideal vs student)

        Phrase-level catches paraphrase cases where a student gives the RIGHT MEANING
        using completely different vocabulary — SBERT full-sentence similarity is low
        but specific key-phrase similarity is high.

        Example:
          Ideal:   "securing information and communication through codes"
          Student: "scrambling a message so only the intended reader can read"
          Full-sentence SBERT: ~0.47 (misses)
          Phrase "securing information" vs student: ~0.55
          Phrase "through the use of codes" vs student "scrambling": ~0.60
          → Best phrase match wins → final score rescued
        """
        details = {}
        full_score = 0.0
        phrase_score = 0.0

        if self.semantic_model is not None:
            try:
                import numpy as _np
                # encode to numpy arrays (works with or without torch)
                ideal_emb   = self.semantic_model.encode(ideal,   convert_to_tensor=False)
                student_emb = self.semantic_model.encode(student, convert_to_tensor=False)

                # Cosine similarity via numpy
                def _cosine(a, b):
                    denom = (_np.linalg.norm(a) * _np.linalg.norm(b))
                    return float(_np.dot(a, b) / denom) if denom > 0 else 0.0

                cos_sim = _cosine(ideal_emb, student_emb)
                full_score = max(0.0, cos_sim) ** 0.6

                # Phrase-level similarity
                import re as _re
                ideal_words_list = ideal.split()
                phrases = []
                for window in (4, 6):
                    for i in range(0, len(ideal_words_list) - window + 1, 2):
                        phrase = ' '.join(ideal_words_list[i:i+window])
                        if len(phrase.split()) >= 3:
                            phrases.append(phrase)
                ideal_sentences = [s.strip() for s in _re.split(r'[.;,]', ideal) if len(s.strip()) > 10]
                phrases.extend(ideal_sentences)

                if phrases:
                    phrase_embs   = self.semantic_model.encode(phrases, convert_to_tensor=False)
                    student_emb_2 = self.semantic_model.encode(student, convert_to_tensor=False)
                    sims = [_cosine(pe, student_emb_2) for pe in phrase_embs]
                    best_phrase_cos = max(sims)
                    phrase_score = max(0.0, best_phrase_cos) ** 0.6
                else:
                    phrase_score = full_score

                final_semantic = max(
                    0.65 * full_score + 0.35 * phrase_score,
                    0.40 * full_score + 0.60 * phrase_score
                )

                details = {
                    'method':            'sentence_transformer + phrase (numpy)',
                    'cosine_similarity': round(cos_sim, 4),
                    'full_score':        round(full_score, 4),
                    'phrase_score':      round(phrase_score, 4),
                    'normalized_score':  round(final_semantic, 4),
                    'model':             'all-MiniLM-L6-v2'
                }
                return final_semantic, details

            except Exception as e:
                print(f"[WARN] SBERT error: {e}")

        # ── Fallback: TF-IDF cosine via sklearn ──
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity as sk_cosine
            vec = TfidfVectorizer(stop_words='english')
            tfidf = vec.fit_transform([ideal, student])
            sim = sk_cosine(tfidf[0], tfidf[1])[0][0]
            return float(sim), {'method': 'tfidf_fallback', 'cosine_similarity': round(sim, 4)}
        except Exception:
            pass

        # ── Last resort: Jaccard ──
        ideal_words   = set(ideal.lower().split()) - STOPWORDS
        student_words = set(student.lower().split()) - STOPWORDS
        if not ideal_words or not student_words:
            return 0.0, {'method': 'jaccard_fallback', 'score': 0.0}
        intersection = ideal_words & student_words
        union = ideal_words | student_words
        jaccard = len(intersection) / len(union)
        return jaccard, {'method': 'jaccard_fallback', 'jaccard_similarity': round(jaccard, 4)}

    # ───────────────────── Layer 3: Structure ────────────────────────────────

    def evaluate_structure(self, answer: str) -> Tuple[float, Dict]:
        """
        Layer 3: Structural coherence.
        Uses ORIGINAL text (not stripped) so sentence boundaries are preserved.
        """
        # Count sentences using original text
        sentences = re.split(r'[.!?]+', answer)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 3]
        sentence_count = len(sentences)

        # Sentence count score
        if 3 <= sentence_count <= 10:
            length_score = 1.0
        elif sentence_count == 2:
            length_score = 0.8
        elif sentence_count == 1:
            length_score = 0.5
        elif sentence_count == 0:
            length_score = 0.0
        else:
            length_score = min(10 / sentence_count, 1.0)

        # Connector words score
        answer_lower = answer.lower()
        connector_count = sum(1 for c in self.connectors if c in answer_lower)
        connector_score = min(connector_count / 3.0, 1.0)

        # Word count score
        words = re.findall(r'\b\w+\b', answer)
        word_count = len(words)
        if 20 <= word_count <= 150:
            word_score = 1.0
        elif word_count < 20:
            word_score = word_count / 20.0
        else:
            word_score = min(150 / word_count, 0.9)

        # Grammar indicators
        capital_starts = sum(1 for s in sentences if s and s[0].isupper())
        grammar_score  = capital_starts / max(sentence_count, 1)

        # Paragraph structure
        paragraphs = [p.strip() for p in re.split(r'\n\n+', answer) if p.strip()]
        para_count = max(len(paragraphs), 1)
        if 1 <= para_count <= 4:
            para_score = 1.0
        else:
            para_score = 0.7

        # Academic style
        academic_hits = sum(1 for t in self.academic_terms if t in answer_lower)
        style_score = min(academic_hits * 0.15, 0.3)

        # Final weighted structure score
        structure_score = (
            0.25 * length_score +
            0.25 * connector_score +
            0.20 * word_score +
            0.15 * grammar_score +
            0.10 * para_score +
            0.05 * style_score
        )

        details = {
            'sentence_count':   sentence_count,
            'word_count':       word_count,
            'connector_count':  connector_count,
            'paragraph_count':  para_count,
            'grammar_score':    round(grammar_score, 2),
            'length_score':     round(length_score, 2),
            'connector_score':  round(connector_score, 2),
            'word_score':       round(word_score, 2),
            'style_score':      round(style_score, 2)
        }

        return min(structure_score, 1.0), details

    # ───────────────────── Layer 4: Completeness ─────────────────────────────

    def evaluate_completeness(self, question: str, student: str, ideal: str) -> Tuple[float, Dict]:
        """
        Layer 4: Completeness — does the student address what the question asks?
        """
        # Keywords from question
        question_keywords = self.extract_keywords(question)

        if not question_keywords:
            return 0.5, {'message': 'No keywords found in question'}

        # Check which question keywords appear in student answer (exact or synonym)
        student_lower = student.lower()
        covered  = []
        missing  = []

        for kw in question_keywords:
            found = kw in student_lower
            if not found:
                # Try synonym
                student_words = set(re.findall(r'\b\w+\b', student_lower))
                found = any(self._synonymous(kw, sw) for sw in student_words)
            if found:
                covered.append(kw)
            else:
                missing.append(kw)

        kw_coverage = len(covered) / len(question_keywords)

        # Question-type specific check
        question_type = self.classify_question(question)
        type_score    = self.evaluate_question_type_coverage(question_type, student, ideal)

        # Ideal answer coverage (how much of ideal content is reflected)
        ideal_words   = set(re.findall(r'\b\w+\b', ideal)) - STOPWORDS
        student_words = set(re.findall(r'\b\w+\b', student_lower)) - STOPWORDS
        ideal_coverage = len(ideal_words & student_words) / len(ideal_words) if ideal_words else 0.0

        completeness_score = (
            0.40 * kw_coverage +
            0.30 * type_score +
            0.30 * ideal_coverage
        )

        details = {
            'question_keywords':  question_keywords,
            'covered_keywords':   covered,
            'missing_keywords':   missing,
            'coverage_percentage': round(kw_coverage * 100, 1),
            'ideal_coverage':     round(ideal_coverage * 100, 1),
            'question_type':      question_type,
            'type_score':         round(type_score, 2)
        }

        return min(completeness_score, 1.0), details

    # ───────────────────── Extraction Helpers ────────────────────────────────

    def extract_concepts(self, text: str, top_n: int = 15) -> List[str]:
        """Extract key concepts using TF-style frequency, filtered with stopwords."""
        words = re.findall(r'\b[a-z]+\b', text.lower())
        filtered = [w for w in words if w not in STOPWORDS and len(w) > 2]
        counts = Counter(filtered)
        concepts = [w for w, _ in counts.most_common(top_n)]
        # Also include academic domain terms found in text
        for term in self.academic_terms:
            if term in text and term not in concepts:
                concepts.append(term)
        return concepts

    def extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from question text."""
        question_words = {
            'what', 'who', 'where', 'when', 'why', 'how', 'explain', 'describe',
            'discuss', 'compare', 'contrast', 'define', 'list', 'state', 'give'
        }
        words = re.findall(r'\b\w+\b', text.lower())
        keywords = [
            w for w in words
            if w not in question_words and w not in STOPWORDS and len(w) > 2
        ]
        return list(dict.fromkeys(keywords))[:8]  # unique, preserve order, top 8

    def calculate_concept_importance(self, text: str, concepts: List[str]) -> Dict[str, float]:
        """TF-based importance scoring with smoothing."""
        words = text.split()
        total = max(len(words), 1)
        importance = {}
        for concept in concepts:
            freq = words.count(concept)
            tf   = freq / total
            importance[concept] = tf + 0.05   # smoothing
        return importance

    def classify_question(self, question: str) -> str:
        """Classify question into a type (definition, explanation, etc.)."""
        ql = question.lower()
        for q_type, patterns in self.question_patterns.items():
            if any(p in ql for p in patterns):
                return q_type
        return 'general'

    def evaluate_question_type_coverage(self, q_type: str, student: str, ideal: str) -> float:
        """Check if answer properly addresses the question type requirements."""
        sl = student.lower()

        if q_type == 'definition':
            if any(p in sl for p in ['is defined as', 'refers to', 'means', 'is a', 'is the']):
                return 1.0
            return 0.7 if len(student.split()) < 40 else 0.6

        elif q_type == 'explanation':
            example_words = ['for example', 'for instance', 'such as', 'like', 'example', 'e.g']
            if any(w in sl for w in example_words):
                return 1.0
            return 0.8 if len(student.split()) > 30 else 0.5

        elif q_type == 'comparison':
            contrast = ['however', 'but', 'although', 'while', 'whereas', 'on the other hand',
                       'in contrast', 'unlike', 'similarly', 'both', 'difference', 'similar']
            hits = sum(1 for w in contrast if w in sl)
            return min(hits * 0.25, 1.0) if hits > 0 else 0.4

        elif q_type in ('advantages', 'disadvantages'):
            points = sl.count('•') + sl.count('-') + sl.count('firstly') + sl.count('first,')
            numbered = len(re.findall(r'\b[1-9]\.\s', student))
            total_points = points + numbered
            if total_points >= 3:
                return 1.0
            elif total_points == 2:
                return 0.8
            elif total_points == 1:
                return 0.6
            return 0.4

        elif q_type == 'process':
            seq = ['first', 'then', 'next', 'after', 'finally', 'step', 'stage', 'subsequently']
            hits = sum(1 for w in seq if w in sl)
            return min(hits * 0.2, 1.0) if hits >= 2 else 0.4 + hits * 0.1

        elif q_type == 'analysis':
            analysis_words = ['because', 'therefore', 'this shows', 'indicates', 'suggests',
                             'demonstrates', 'evidence', 'critically', 'evaluate']
            hits = sum(1 for w in analysis_words if w in sl)
            return min(0.5 + hits * 0.1, 1.0)

        return 0.7   # general

    # ───────────────────── Weights ────────────────────────────────────────────

    def get_dynamic_weights(self, question_type: str) -> List[float]:
        """
        Return layer weights based on question type.
        [conceptual, semantic, structural, completeness]

        For DEFINITION and EXPLANATION questions we weight SEMANTIC (SBERT) very
        highly because the student may correctly answer in their own words without
        using the exact keyword vocabulary from the ideal answer.
        """
        weight_map = {
            # SBERT dominates: definitions are about meaning, not keywords
            'definition':    [0.25, 0.55, 0.10, 0.10],
            # Strong SBERT weight: explanations also rely on meaning
            'explanation':   [0.25, 0.45, 0.15, 0.15],
            'comparison':    [0.25, 0.35, 0.20, 0.20],
            'advantages':    [0.30, 0.25, 0.25, 0.20],
            'disadvantages': [0.30, 0.25, 0.25, 0.20],
            # Process: structure matters most
            'process':       [0.20, 0.30, 0.30, 0.20],
            'analysis':      [0.25, 0.35, 0.20, 0.20],
            'general':       [0.25, 0.40, 0.20, 0.15]
        }
        return weight_map.get(question_type, weight_map['general'])

    # ───────────────────── Feedback Generation ───────────────────────────────

    def generate_feedback(self, concept_score, semantic_score, structure_score,
                          completeness_score, concept_details, completeness_details,
                          accuracy_details=None) -> Tuple[str, Dict]:
        """
        Generate specific, actionable, student-friendly feedback.
        Returns: (simple_string_feedback, detailed_dict_feedback)
        """
        parts = []
        detailed = {}

        # ── Concept feedback ──
        missed = concept_details.get('missed_concepts', [])
        syn_pairs = concept_details.get('synonym_pairs', [])
        if concept_score < 0.5:
            msg = "❌ Key concepts missing"
            if missed:
                msg += f": '{', '.join(missed[:3])}'. Add these to your answer."
            parts.append(msg)
            detailed['concepts'] = {'status': 'weak', 'missed': missed[:5], 'tip': 'Study the core definitions and include subject-specific terminology.'}
        elif concept_score < 0.75:
            msg = "⚠ Most concepts covered"
            if missed:
                msg += f", but try to add: '{missed[0]}'."
            parts.append(msg)
            detailed['concepts'] = {'status': 'partial', 'missed': missed[:3]}
        else:
            parts.append("✅ Excellent concept coverage.")
            detailed['concepts'] = {'status': 'good'}

        # Note synonym matches (positive reinforcement)
        if syn_pairs:
            detailed['synonym_recognition'] = {
                'message': f"System recognized {len(syn_pairs)} equivalent term(s) you used.",
                'examples': [f"'{p[0]}' ≈ '{p[1]}'" for p in syn_pairs[:3]]
            }

        # ── Semantic feedback ──
        if semantic_score < 0.4:
            parts.append("❌ Answer meaning is quite different from expected — re-read the question carefully.")
            detailed['semantic'] = {'status': 'weak', 'tip': 'Make sure your answer addresses what the question is actually asking.'}
        elif semantic_score < 0.65:
            parts.append("⚠ Partially on topic — expand your explanation.")
            detailed['semantic'] = {'status': 'partial', 'tip': 'Add more relevant details and evidence.'}
        else:
            detailed['semantic'] = {'status': 'good'}

        # ── Structure feedback ──
        if structure_score < 0.5:
            parts.append("❌ Improve structure: Use proper sentences, connecting words (therefore, however, firstly), and organize into paragraphs.")
            detailed['structure'] = {'status': 'weak', 'tip': 'Use 3–8 sentences. Start a new paragraph for each main idea.'}
        elif structure_score < 0.7:
            parts.append("⚠ Structure could be improved — add connecting words like 'therefore', 'furthermore', 'for example'.")
            detailed['structure'] = {'status': 'partial'}
        else:
            detailed['structure'] = {'status': 'good'}

        # ── Completeness feedback ──
        missing_kw = completeness_details.get('missing_keywords', [])
        cov = completeness_details.get('coverage_percentage', 0)
        if completeness_score < 0.5:
            parts.append(f"❌ Answer is incomplete ({cov:.0f}% coverage). Address: {', '.join(missing_kw[:3]) if missing_kw else 'all parts of the question'}.")
            detailed['completeness'] = {'status': 'weak', 'missing': missing_kw[:4]}
        elif completeness_score < 0.75:
            if missing_kw:
                parts.append(f"⚠ Almost complete — try to mention '{missing_kw[0]}'.")
            detailed['completeness'] = {'status': 'partial', 'missing': missing_kw[:2]}
        else:
            detailed['completeness'] = {'status': 'good'}

        # ── Accuracy engine hints ──
        if accuracy_details and accuracy_details.get('feedback_hints'):
            for h in accuracy_details['feedback_hints']:
                if h not in parts and '✓' in h:  # add positive hints
                    parts.append(h)

        # ── Overall positive ──
        overall = (concept_score + semantic_score + structure_score + completeness_score) / 4
        if overall >= 0.80:
            parts.insert(0, "🌟 Great answer overall!")
        elif not parts:
            parts.append("Good effort. Keep practising for better results.")

        return " | ".join(parts), detailed

    # ───────────────────── Confidence ────────────────────────────────────────

    def calculate_confidence(self, scores: List[float]) -> float:
        """Calculate confidence in the evaluation result."""
        avg = float(np.mean(scores))
        var = float(np.var(scores))

        if var < 0.02:
            consistency = 0.9
        elif var < 0.05:
            consistency = 0.75
        elif var < 0.10:
            consistency = 0.55
        else:
            consistency = 0.35

        confidence = (avg * 0.6 + consistency * 0.4)
        return round(confidence * 100, 1)


# ─────────────────────────── Test Function ────────────────────────────────────

def test_evaluator():
    """Test with real college-style questions including synonym cases."""
    print("\n" + "="*65)
    print("TESTING ADVANCED ANSWER EVALUATOR (Production Grade)")
    print("="*65)

    evaluator = AdvancedAnswerEvaluator()

    cases = [
        {
            "label": "Synonym test: automobile = car",
            "question": "What is an automobile?",
            "ideal":   "An automobile is a motorized vehicle used for transportation on roads.",
            "student": "A car is a machine powered by an engine used to travel on roads.",
            "expect":  ">= 65"
        },
        {
            "label": "High similarity — correct answer",
            "question": "What are the advantages of renewable energy?",
            "ideal":   "Renewable energy reduces greenhouse gases, is sustainable, lowers long-term costs, and creates jobs.",
            "student": "Renewable energy is eco-friendly, sustainable, and creates employment. It also reduces pollution and saves money.",
            "expect":  ">= 75"
        },
        {
            "label": "Wrong answer — should score LOW",
            "question": "What affects plant growth?",
            "ideal":   "Plants need sunlight, water, and nutrients to grow well.",
            "student": "Animals grow fast when they eat meat and fish regularly.",
            "expect":  "<= 35"
        },
        {
            "label": "Empty answer — should score 0",
            "question": "Explain gravity",
            "ideal":   "Gravity is the force that attracts objects with mass toward each other.",
            "student": "  ",
            "expect":  "== 0"
        }
    ]

    print()
    all_ok = True
    for c in cases:
        result = evaluator.evaluate(c['question'], c['ideal'], c['student'])
        score  = result['final_score']
        grade  = result['grade']
        exp    = c['expect']

        # Check expectation
        if '>=' in exp:
            passed = score >= int(exp.split('>=')[1].strip())
        elif '<=' in exp:
            passed = score <= int(exp.split('<=')[1].strip())
        else:
            passed = score == 0

        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {c['label']}: Score={score}/100  Grade={result['grade']}")

    print("="*65)
    print("ALL TESTS PASSED" if all_ok else "SOME TESTS FAILED")
    print("="*65)


if __name__ == "__main__":
    test_evaluator()