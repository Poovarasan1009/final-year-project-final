"""
ACCURACY ENGINE - Synonym-Aware, N-gram & Semantic Boosting
Handles the core problem: "same meaning, different words"
e.g., "automobile" == "car", "purchase" == "buy", "rapid" == "fast"
"""
import re
import math
from typing import Dict, List, Tuple, Set
from collections import Counter

# ── Manual Synonym Table ─────────────────────────────────────────────────────
# WordNet misses many common word pairs; this table fills those gaps.
# Built as a symmetric mapping: each word maps to its synonyms.
_MANUAL_SYNONYMS: Dict[str, Set[str]] = {}

def _build_manual_synonyms():
    """Build bidirectional synonym mapping from raw synonym groups."""
    raw_groups = [
        # Speed / intensity
        {'fast', 'rapid', 'quick', 'swift', 'speedy', 'brisk', 'hasty'},
        {'slow', 'gradual', 'sluggish', 'leisurely'},
        # Ecology / environment
        {'eco-friendly', 'sustainable', 'green', 'renewable', 'environmentally friendly'},
        {'environment', 'nature', 'ecosystem', 'ecology'},
        {'pollution', 'contamination', 'degradation'},
        # Economy / finance
        {'cost', 'expense', 'price', 'fee', 'charge'},
        {'cheap', 'inexpensive', 'affordable', 'economical', 'low-cost'},
        {'expensive', 'costly', 'pricey', 'high-cost'},
        {'save', 'reduce', 'lower', 'decrease', 'cut', 'minimise', 'minimize'},
        {'revenue', 'income', 'earnings', 'profit', 'return'},
        {'purchase', 'buy', 'acquire', 'obtain', 'procure'},
        # Work / jobs
        {'job', 'employment', 'work', 'occupation', 'career', 'vocation'},
        {'employee', 'worker', 'staff', 'personnel', 'workforce'},
        {'create', 'generate', 'produce', 'develop', 'make', 'form'},
        # Core science verbs
        {'use', 'utilise', 'utilize', 'employ', 'apply', 'leverage'},
        {'show', 'demonstrate', 'indicate', 'reveal', 'display', 'illustrate'},
        {'help', 'assist', 'aid', 'support', 'facilitate'},
        {'need', 'require', 'demand', 'necessitate'},
        {'improve', 'enhance', 'boost', 'upgrade', 'strengthen', 'advance'},
        {'reduce', 'decrease', 'diminish', 'lower', 'lessen', 'minimise', 'minimize'},
        {'increase', 'grow', 'rise', 'expand', 'augment', 'boost'},
        {'change', 'alter', 'modify', 'transform', 'shift'},
        {'cause', 'result in', 'lead to', 'produce', 'generate'},
        # Common academic adjectives
        {'important', 'significant', 'crucial', 'vital', 'essential', 'key'},
        {'complex', 'complicated', 'intricate', 'difficult', 'involved'},
        {'simple', 'easy', 'straightforward', 'basic', 'elementary'},
        {'large', 'big', 'great', 'major', 'substantial', 'sizeable'},
        {'small', 'little', 'minor', 'tiny', 'minimal'},
        {'many', 'numerous', 'multiple', 'several', 'various', 'diverse'},
        # Transport / movement
        {'automobile', 'car', 'vehicle', 'motorcar', 'auto'},
        {'travel', 'move', 'transport', 'commute', 'transit'},
        {'road', 'street', 'highway', 'route', 'path'},
        # Energy / power
        {'energy', 'power', 'electricity', 'fuel'},
        {'powered', 'driven', 'operated', 'run', 'fuelled', 'fueled'},
        {'emit', 'release', 'discharge', 'produce', 'generate'},
        # Biology
        {'plant', 'vegetation', 'flora', 'greenery', 'crop'},
        {'grow', 'develop', 'thrive', 'flourish', 'cultivate'},
        {'sun', 'sunlight', 'solar', 'light', 'radiation'},
        {'water', 'moisture', 'liquid', 'fluid', 'hydration'},
        {'animal', 'creature', 'organism', 'beast', 'fauna'},
        # CS / tech / ML
        {'algorithm', 'method', 'procedure', 'process', 'approach', 'technique', 'way', 'strategy'},
        {'computer', 'machine', 'system', 'device', 'processor', 'hardware', 'equipment'},
        {'data', 'information', 'content', 'records', 'figures', 'statistics'},
        {'store', 'save', 'retain', 'preserve', 'keep', 'memorise', 'memorize'},
        {'machine learning', 'ml', 'ai', 'artificial intelligence', 'automated learning'},
        {'predict', 'forecast', 'projection', 'anticipation', 'forecasted', 'forecasting'},
        {'outcome', 'result', 'consequence', 'output', 'conclusion', 'finding'},
        # Communication
        {'say', 'state', 'mention', 'note', 'describe', 'explain', 'express', 'convey', 'detail'},
        {'find', 'discover', 'identify', 'detect', 'determine', 'recognise', 'recognize'},
        {'complete', 'finish', 'conclude', 'end', 'finalise', 'finalize', 'achieve', 'attain'},
        # Measurement
        {'measure', 'assess', 'evaluate', 'quantify', 'gauge', 'appraise'},
        {'test', 'examine', 'verify', 'check', 'validate', 'assess', 'audit'},
        # Positive descriptors
        {'correct', 'accurate', 'precise', 'exact', 'right', 'valid', 'proper'},
        {'effective', 'efficient', 'productive', 'successful', 'optimal', 'ideal'},
        {'benefit', 'advantage', 'merit', 'gain', 'profit', 'strength', 'plus'},
        {'disadvantage', 'drawback', 'limitation', 'weakness', 'constraint', 'minus'},
        # Structure / form
        {'structure', 'framework', 'organisation', 'organization', 'arrangement', 'layout', 'design'},
        {'part', 'component', 'element', 'section', 'segment', 'portion', 'unit'},
        # Comparison
        {'same', 'identical', 'equivalent', 'equal', 'matching', 'uniform'},
        {'different', 'diverse', 'various', 'varied', 'distinct', 'unique', 'separate'},
    ]
    for group in raw_groups:
        for word in group:
            if word not in _MANUAL_SYNONYMS:
                _MANUAL_SYNONYMS[word] = set()
            _MANUAL_SYNONYMS[word] |= (group - {word})

_build_manual_synonyms()


# ── Synonym-Aware Matching via WordNet ─────────────────────────────────────────
def get_synonyms(word: str) -> Set[str]:
    """
    Get all synonyms of a word: first checks manual table, then WordNet.
    Returns a set of lower-cased synonym strings.
    """
    word_l = word.lower()
    synonyms = {word_l}

    # Manual table (faster, handles WordNet gaps)
    synonyms |= _MANUAL_SYNONYMS.get(word_l, set())

    # WordNet (broader coverage)
    try:
        from nltk.corpus import wordnet
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                name = lemma.name().lower().replace('_', ' ')
                synonyms.add(name)
    except Exception:
        pass
    return synonyms


def words_are_synonymous(word1: str, word2: str) -> bool:
    """
    Return True if word1 and word2 are synonyms.
    Checks: exact match → manual table → WordNet synsets.
    Examples:
      "rapid"      ↔ "fast"        → True (manual table)
      "automobile" ↔ "car"         → True (manual table + WordNet)
      "purchase"   ↔ "buy"         → True (manual table + WordNet)
      "happy"      ↔ "planet"      → False
    """
    w1, w2 = word1.lower(), word2.lower()
    if w1 == w2:
        return True
    # Check manual table (bidirectional)
    if w2 in _MANUAL_SYNONYMS.get(w1, set()):
        return True
    if w1 in _MANUAL_SYNONYMS.get(w2, set()):
        return True
    # Check WordNet synset overlap
    try:
        from nltk.corpus import wordnet
        synsets1 = set(wordnet.synsets(word1))
        synsets2 = set(wordnet.synsets(word2))
        return bool(synsets1 & synsets2)
    except Exception:
        return False


# ── Concept-Phrase Paraphrase Groups ─────────────────────────────────────────
# These go beyond single-word synonyms to PHRASE-level equivalence.
# Handles the hardest paraphrase cases where SBERT also fails, e.g.:
#   "securing information via codes"  ≡  "scrambling readable messages"
#   "primary key uniquely identifies" ≡  "unique identifier for each row"
#
# Format: {canonical_concept: [paraphrase_phrase_1, paraphrase_phrase_2, ...]}
# Each phrase is checked as a SUBSTRING in the student answer (lowercased).
CONCEPT_PHRASES: Dict[str, List[str]] = {
    # Cryptography / Security
    'securing information':       ['scrambl', 'encrypt', 'cipher', 'hidden writing',
                                    'protect information', 'secure data', 'lock message',
                                    'secret message', 'hide message', 'code message'],
    'codes':                      ['cipher', 'encrypt', 'scramble', 'hidden writing',
                                    'secret code', 'encode', 'coded'],
    'unauthorized access':        ['only intended', 'only recipient', 'only the person',
                                    'cannot be read', 'unreadable', 'private',
                                    'prevent access', 'keep secret'],
    'communication':              ['message', 'information', 'data', 'signal',
                                    'transmit', 'send', 'exchange'],

    # Database / CS
    'primary key':                ['unique identifier', 'unique id', 'uniquely identifies',
                                    'identify each row', 'unique column'],
    'unique identifier':          ['primary key', 'unique id', 'unique key'],
    'foreign key':                ['reference', 'links two tables', 'relationship between tables'],
    'normalization':              ['eliminate redundancy', 'reduce duplication', 'organize data'],
    'sql':                        ['structured query language', 'database language', 'query language'],
    'algorithm':                  ['step-by-step process', 'procedure', 'set of instructions',
                                    'method to solve', 'sequence of steps'],
    'recursion':                  ['function calls itself', 'self-referential', 'calls itself',
                                    'base case', 'recursive call'],
    'object oriented':            ['oop', 'class and object', 'classes and objects',
                                    'encapsulation inheritance polymorphism'],
    'inheritance':                ['child class', 'subclass', 'parent class', 'base class',
                                    'derives from', 'extends', 'inherits from'],
    'polymorphism':               ['same method different behavior', 'method overriding',
                                    'method overloading', 'one interface many forms'],
    'encapsulation':              ['data hiding', 'hide internal data', 'private data',
                                    'wrap data and methods'],
    'data structure':             ['way to organize data', 'store and organize', 'collection of data'],
    'linked list':                ['nodes connected', 'pointer to next', 'chain of nodes'],
    'stack':                      ['last in first out', 'lifo', 'push and pop'],
    'queue':                      ['first in first out', 'fifo', 'enqueue dequeue'],
    'binary search':              ['divide in half', 'halve the search', 'sorted array'],
    'machine learning':           ['computers learn from data', 'algorithms learn patterns',
                                    'train on examples', 'learn from examples', 'model trained'],
    'artificial intelligence':    ['machines that think', 'simulate human intelligence',
                                    'computers perform tasks', 'intelligent machines'],
    'operating system':           ['manages hardware', 'system software', 'manages resources',
                                    'interface between hardware and software'],
    'cloud computing':            ['computing over internet', 'remote servers', 'internet-based computing',
                                    'on-demand resources', 'hosted services'],

    # Biology
    'photosynthesis':             ['convert sunlight to food', 'light to chemical energy',
                                    'plants make food', 'glucose from sunlight', 'chlorophyll absorbs'],
    'dna':                        ['genetic material', 'genetic code', 'hereditary information',
                                    'carries genes', 'double helix'],
    'cell division':              ['mitosis', 'meiosis', 'cell splits', 'reproduce cells'],
    'ecosystem':                  ['living organisms and environment', 'plants animals environment',
                                    'biotic and abiotic'],
    'natural selection':          ['survival of fittest', 'best adapted survive',
                                    'darwin', 'favorable traits pass on'],

    # Physics
    'newtons first law':          ['object at rest stays at rest', 'inertia', 'no net force',
                                    'constant velocity', 'unless acted upon'],
    'kinetic energy':             ['energy of motion', 'moving objects have energy',
                                    'half mv squared', 'energy due to movement'],
    'potential energy':           ['stored energy', 'energy due to position',
                                    'energy at height', 'gravitational potential'],
    'electric current':           ['flow of charge', 'flow of electrons', 'amperes', 'charge carriers'],
    'electromagnetic wave':       ['light wave', 'radio wave', 'em wave', 'oscillating fields'],

    # Mathematics
    'derivative':                 ['rate of change', 'slope of tangent', 'differential',
                                    'instantaneous rate', 'dy by dx'],
    'integration':                ['area under curve', 'antiderivative', 'sum of infinitesimal',
                                    'total from rate'],
    'probability':                ['likelihood', 'chance', 'likelihood of event',
                                    'outcomes divided by total'],

    # Economics
    'inflation':                  ['rise in prices', 'prices increase', 'purchasing power decreases',
                                    'value of money falls', 'cost of living rises'],
    'supply and demand':          ['market forces', 'price mechanism', 'equilibrium price',
                                    'quantity demanded', 'quantity supplied'],
    'gdp':                        ['gross domestic product', 'total economic output',
                                    'value of goods and services', 'national income'],
}


def concept_phrase_overlap(ideal_text: str, student_text: str) -> Dict:
    """
    Check how many of the ideal answer's domain concepts the student answer covers,
    using CONCEPT_PHRASES to recognize paraphrase expressions.

    Returns:
      - overlap_score: float [0, 1]
      - matched_concepts: list of concept names that were found
      - missed_concepts: list of concept names not found
    """
    student_lower  = student_text.lower()
    ideal_lower    = ideal_text.lower()

    # Find which canonical concepts appear in the ideal answer
    relevant = []
    for concept, phrases in CONCEPT_PHRASES.items():
        # Check if concept name OR any of its phrases appear in ideal
        if concept in ideal_lower or any(p in ideal_lower for p in phrases):
            relevant.append(concept)

    if not relevant:
        return {'overlap_score': 0.0, 'matched_concepts': [], 'missed_concepts': []}

    matched  = []
    missed   = []
    for concept in relevant:
        phrases = CONCEPT_PHRASES[concept]
        # Check if student has the concept name itself
        found = concept in student_lower
        # Or any of its paraphrase phrases
        if not found:
            found = any(p in student_lower for p in phrases)
        # Or the concept name itself (partial match)
        if not found:
            found = any(word in student_lower for word in concept.split() if len(word) > 4)

        if found:
            matched.append(concept)
        else:
            missed.append(concept)

    overlap_score = len(matched) / len(relevant) if relevant else 0.0
    return {
        'overlap_score':    overlap_score,
        'matched_concepts': matched,
        'missed_concepts':  missed,
        'total_concepts':   len(relevant),
    }




# ── N-gram Helpers ─────────────────────────────────────────────────────────────
def get_ngrams(text: str, n: int) -> Counter:
    """Extract n-grams from cleaned text."""
    tokens = re.findall(r'\b[a-z]+\b', text.lower())
    ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
    return Counter(ngrams)


def ngram_precision_recall(reference: str, hypothesis: str, n: int = 1) -> Tuple[float, float]:
    """
    Compute n-gram precision and recall between reference and hypothesis.
    Precision = how much of hypothesis appears in reference (no hallucination)
    Recall    = how much of reference is covered by hypothesis (completeness)
    """
    ref_ngrams    = get_ngrams(reference, n)
    hyp_ngrams    = get_ngrams(hypothesis, n)

    if not hyp_ngrams or not ref_ngrams:
        return 0.0, 0.0

    # Clipped matches (reference-capped)
    clipped = sum(min(count, ref_ngrams[gram]) for gram, count in hyp_ngrams.items())
    hyp_total = sum(hyp_ngrams.values())
    ref_total = sum(ref_ngrams.values())

    precision = clipped / hyp_total if hyp_total > 0 else 0.0
    recall    = clipped / ref_total if ref_total > 0 else 0.0

    return precision, recall


def bleu_style_score(reference: str, hypothesis: str,
                     max_n: int = 2, brevity_penalty: bool = True) -> float:
    """
    BLEU-style score (unigram + bigram).
    Penalises very short answers (brevity penalty).
    Returns score in [0, 1].
    """
    scores = []
    for n in range(1, max_n + 1):
        p, _ = ngram_precision_recall(reference, hypothesis, n)
        scores.append(p)

    if not scores or all(s == 0 for s in scores):
        return 0.0

    # Geometric mean of precisions (clip zeros with small epsilon)
    log_avg = sum(math.log(max(s, 1e-9)) for s in scores) / len(scores)
    bleu = math.exp(log_avg)

    if brevity_penalty:
        ref_len = len(re.findall(r'\b\w+\b', reference))
        hyp_len = len(re.findall(r'\b\w+\b', hypothesis))
        if hyp_len < ref_len and ref_len > 0:
            bp = math.exp(1 - ref_len / max(hyp_len, 1))
            bleu *= bp

    return min(bleu, 1.0)


# ── Synonym-Expanded Keyword Overlap ──────────────────────────────────────────
STOPWORDS = {
    'a', 'an', 'the', 'and', 'or', 'but', 'is', 'are', 'was', 'were',
    'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
    'will', 'would', 'could', 'should', 'may', 'might', 'shall', 'can',
    'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from', 'as',
    'into', 'through', 'during', 'before', 'after', 'above', 'below',
    'this', 'that', 'these', 'those', 'it', 'its', 'we', 'they', 'he',
    'she', 'you', 'i', 'me', 'him', 'her', 'us', 'them', 'my', 'our',
    'your', 'his', 'their', 'which', 'who', 'whom', 'what', 'when',
    'where', 'how', 'if', 'than', 'so', 'such', 'not', 'no', 'more',
    'also', 'just', 'very', 'only', 'even', 'get', 'got', 'make', 'made',
    'about', 'each', 'both', 'all', 'any', 'some', 'most', 'other'
}


def extract_content_words(text: str) -> List[str]:
    """Extract meaningful content words, filtering stopwords."""
    tokens = re.findall(r'\b[a-z]+\b', text.lower())
    return [t for t in tokens if t not in STOPWORDS and len(t) > 2]


def synonym_aware_overlap(ideal_text: str, student_text: str, use_wordnet: bool = True) -> Dict:
    """
    The Key Feature: Synonym-aware keyword matching.

    Instead of exact string matching, checks if any word in the student's
    answer is SYNONYMOUS with each keyword from the ideal answer.

    Example:
      Ideal:   "An automobile is a motorized vehicle"
      Student: "A car is a motor-powered machine"
      → 'automobile' → synonyms include 'car'  ✓ MATCH
      → 'motorized'  → synonyms include 'powered' ✓ NEAR-MATCH
      → Score: HIGH (not a mismatch!)

    Returns dict with:
      - matched_words: list of ideal keywords that were found (exact or via synonym)
      - missed_words: list of ideal keywords not found in student answer
      - synonym_matches: list of (ideal_word, student_word) synonym pairs detected
      - overlap_score: float [0, 1]
    """
    ideal_words   = extract_content_words(ideal_text)
    student_words = extract_content_words(student_text)
    student_set   = set(student_words)

    matched_words   = []
    missed_words    = []
    synonym_matches = []

    for iw in set(ideal_words):   # use set to deduplicate
        # 1. Exact match
        if iw in student_set:
            matched_words.append(iw)
            continue

        if use_wordnet:
            # 2. Stem-based match (prefix of 5+ chars)
            prefix_matched = False
            if len(iw) >= 5:
                for sw in student_set:
                    if len(sw) >= 5 and (iw[:5] == sw[:5]):
                        matched_words.append(iw)
                        synonym_matches.append((iw, sw, 'stem'))
                        prefix_matched = True
                        break
            if prefix_matched:
                continue

            # 3. Synonym match via WordNet
            found_syn = False
            iw_synonyms = get_synonyms(iw)
            for sw in student_set:
                # Direct synonym set intersection
                if sw in iw_synonyms:
                    matched_words.append(iw)
                    synonym_matches.append((iw, sw, 'wordnet_synonym'))
                    found_syn = True
                    break
                # Bidirectional synset overlap (more thorough)
                if words_are_synonymous(iw, sw):
                    matched_words.append(iw)
                    synonym_matches.append((iw, sw, 'wordnet_synset'))
                    found_syn = True
                    break
            if found_syn:
                continue

        missed_words.append(iw)

    total_ideal = len(set(ideal_words))
    overlap_score = len(matched_words) / total_ideal if total_ideal > 0 else 0.0

    return {
        'matched_words':   matched_words,
        'missed_words':    missed_words,
        'synonym_matches': synonym_matches,
        'overlap_score':   overlap_score,
        'total_ideal_kw':  total_ideal,
        'match_count':     len(matched_words)
    }


# ── Subject Domain Keyword Boosting ───────────────────────────────────────────
DOMAIN_KEYWORDS: Dict[str, List[str]] = {
    'computer_science': [
        'algorithm', 'data', 'structure', 'complexity', 'binary', 'recursion',
        'neural', 'network', 'machine', 'learning', 'artificial', 'intelligence',
        'database', 'query', 'indexed', 'encryption', 'protocol', 'compiler',
        'syntax', 'runtime', 'memory', 'cache', 'thread', 'process', 'kernel',
        'object', 'class', 'inheritance', 'polymorphism', 'encapsulation',
        'supervised', 'unsupervised', 'reinforcement', 'gradient', 'backpropagation'
    ],
    'biology': [
        'cell', 'dna', 'rna', 'protein', 'enzyme', 'photosynthesis', 'mitosis',
        'meiosis', 'chromosome', 'gene', 'mutation', 'evolution', 'organism',
        'metabolism', 'respiration', 'osmosis', 'diffusion', 'membrane',
        'nucleus', 'cytoplasm', 'chlorophyll', 'chloroplast', 'mitochondria'
    ],
    'physics': [
        'force', 'velocity', 'acceleration', 'momentum', 'energy', 'mass',
        'gravity', 'friction', 'thermodynamics', 'entropy', 'quantum', 'wave',
        'frequency', 'amplitude', 'voltage', 'current', 'resistance', 'magnetic',
        'electric', 'nuclear', 'fission', 'fusion', 'relativity', 'inertia'
    ],
    'chemistry': [
        'atom', 'molecule', 'element', 'compound', 'reaction', 'bond', 'ionic',
        'covalent', 'electron', 'proton', 'neutron', 'valence', 'periodic',
        'oxidation', 'reduction', 'acid', 'base', 'ph', 'catalyst', 'equilibrium'
    ],
    'mathematics': [
        'theorem', 'proof', 'equation', 'derivative', 'integral', 'matrix',
        'vector', 'probability', 'statistics', 'function', 'limit', 'series',
        'convergence', 'differential', 'polynomial', 'logarithm', 'exponential'
    ],
    'economics': [
        'supply', 'demand', 'market', 'price', 'inflation', 'gdp', 'fiscal',
        'monetary', 'interest', 'capital', 'investment', 'trade', 'deficit',
        'surplus', 'elasticity', 'equilibrium', 'monopoly', 'competition'
    ],
    'general': [
        'hypothesis', 'theory', 'analysis', 'conclusion', 'methodology',
        'variable', 'constant', 'quantitative', 'qualitative', 'sustainable',
        'environmental', 'economic', 'efficiency', 'optimization'
    ]
}


def get_domain_boost(ideal_text: str, student_text: str, subject: str = None) -> float:
    """
    Give a small accuracy boost when student uses domain-specific terminology
    from the correct subject.
    Returns a boost value [0.0, 0.15] to be added to the final score.
    """
    subject_key = None
    if subject:
        sl = subject.lower()
        if 'computer' in sl or 'programming' in sl or 'software' in sl or 'cs' in sl:
            subject_key = 'computer_science'
        elif 'bio' in sl:
            subject_key = 'biology'
        elif 'phys' in sl:
            subject_key = 'physics'
        elif 'chem' in sl:
            subject_key = 'chemistry'
        elif 'math' in sl:
            subject_key = 'mathematics'
        elif 'econ' in sl:
            subject_key = 'economics'

    # Detect domain from ideal answer keywords if subject not given
    if not subject_key:
        ideal_lower = ideal_text.lower()
        best_domain = 'general'
        best_count = 0
        for domain, kws in DOMAIN_KEYWORDS.items():
            cnt = sum(1 for kw in kws if kw in ideal_lower)
            if cnt > best_count:
                best_count = cnt
                best_domain = domain
        subject_key = best_domain

    domain_kws = DOMAIN_KEYWORDS.get(subject_key, DOMAIN_KEYWORDS['general'])
    student_lower = student_text.lower()

    domain_hits = sum(1 for kw in domain_kws if kw in student_lower)
    boost = min(domain_hits * 0.02, 0.15)   # max 15% boost, 2% per keyword
    return boost


# ── Main Accuracy Engine ──────────────────────────────────────────────────────
class AccuracyEngine:
    """
    Combines all accuracy methods:
      1. BLEU-style n-gram precision/recall
      2. Synonym-aware keyword overlap (the key fix for same-meaning-different-word)
      3. Domain keyword boosting
    Returns a comprehensive accuracy report.
    """

    def __init__(self, use_wordnet: bool = True):
        self.use_wordnet = use_wordnet
        # Try NLTK download for first run
        self._ensure_nltk()

    def _ensure_nltk(self):
        """Download required NLTK data silently."""
        try:
            import nltk
            import os
            nltk_data_dir = os.path.join(os.path.expanduser("~"), "nltk_data")
            for resource in ['wordnet', 'stopwords', 'omw-1.4']:
                try:
                    nltk.data.find(f'corpora/{resource}')
                except LookupError:
                    try:
                        nltk.download(resource, quiet=True, download_dir=nltk_data_dir)
                    except Exception:
                        pass
        except Exception:
            pass

    def score(self, question: str, ideal_answer: str, student_answer: str,
              subject: str = None) -> Dict:
        """
        Main entry point. Returns:
          - accuracy_score: float [0, 1] — combined accuracy
          - synonym_boost:  float [0, 1] — bonus from synonym matching
          - domain_boost:   float [0, 1] — bonus from domain keywords
          - bleu_score:     float [0, 1] — n-gram precision score
          - recall_score:   float [0, 1] — how much of ideal is covered
          - synonym_matches: list of (ideal_word, student_word, match_type)
          - missed_keywords: list of keywords missing from student answer
          - feedback_hints:  list of feedback strings
        """
        # Guard: empty answers
        if not student_answer or len(student_answer.strip()) < 5:
            return {
                'accuracy_score': 0.0,
                'synonym_boost': 0.0,
                'domain_boost': 0.0,
                'bleu_score': 0.0,
                'recall_score': 0.0,
                'synonym_matches': [],
                'missed_keywords': [],
                'feedback_hints': ['Answer is too short or empty.']
            }

        # 1. BLEU-style n-gram score
        bleu = bleu_style_score(ideal_answer, student_answer, max_n=2)

        # 2. N-gram recall (completeness from reference side)
        _, recall_1 = ngram_precision_recall(ideal_answer, student_answer, n=1)

        # 3. Synonym-aware overlap (THE key feature)
        syn_result = synonym_aware_overlap(ideal_answer, student_answer,
                                           use_wordnet=self.use_wordnet)
        syn_overlap = syn_result['overlap_score']
        synonym_matches = syn_result['synonym_matches']
        missed_kw = syn_result['missed_words']

        # 4. Domain boost
        domain_boost = get_domain_boost(ideal_answer, student_answer, subject)

        # 5. Concept-phrase paraphrase overlap (domain dictionary)
        #    The strongest signal for "same meaning, different vocabulary"
        cpo_result   = concept_phrase_overlap(ideal_answer, student_answer)
        cpo_score    = cpo_result.get('overlap_score', 0.0)

        # 6. Combine all signals
        #    Concept-phrase and synonym overlap get highest weight
        #    (they directly solve the paraphrase problem)
        if cpo_result.get('total_concepts', 0) > 0:
            # Domain concepts were found in ideal — blend concept-phrase in heavily
            combined = (
                0.20 * bleu +
                0.15 * recall_1 +
                0.35 * syn_overlap +   # synonym-aware word overlap
                0.30 * cpo_score       # domain concept-phrase overlap
            )
        else:
            # No domain concepts in ideal — fall back to typical weights
            combined = (
                0.30 * bleu +
                0.25 * recall_1 +
                0.45 * syn_overlap
            )

        accuracy = min(combined + domain_boost, 1.0)

        # 7. Feedback hints
        hints = self._generate_hints(syn_result, bleu, domain_boost)

        return {
            'accuracy_score':       round(accuracy, 4),
            'synonym_boost':        round(syn_overlap, 4),
            'domain_boost':         round(domain_boost, 4),
            'bleu_score':           round(bleu, 4),
            'recall_score':         round(recall_1, 4),
            'synonym_overlap':      round(syn_overlap, 4),
            'concept_phrase_score': round(cpo_score, 4),
            'concept_matches':      cpo_result.get('matched_concepts', []),
            'synonym_matches':      synonym_matches,
            'missed_keywords':      missed_kw[:5],
            'matched_keywords':     syn_result['matched_words'],
            'feedback_hints':       hints
        }

    def _generate_hints(self, syn_result: Dict, bleu: float, domain_boost: float) -> List[str]:
        hints = []
        missed = syn_result.get('missed_words', [])
        syn_matches = syn_result.get('synonym_matches', [])
        overlap = syn_result.get('overlap_score', 0.0)

        if overlap < 0.3:
            if missed:
                hints.append(f"Key concepts missing: {', '.join(missed[:4])}. Include these in your answer.")
            else:
                hints.append("Answer doesn't cover enough key ideas from the expected answer.")

        elif overlap < 0.6:
            if missed:
                hints.append(f"Partially good. Try including: {', '.join(missed[:3])}.")

        if bleu < 0.2:
            hints.append("Use more specific terms and phrases from the subject domain.")

        if syn_matches:
            # Positive reinforcement for synonym usage
            ex = syn_matches[0]
            hints.append(f"✓ Good use of equivalent term '{ex[1]}' for '{ex[0]}'.")

        if domain_boost > 0.05:
            hints.append("✓ Correct use of subject-specific terminology.")

        if not hints:
            hints.append("Excellent! Answer covers all key concepts correctly.")

        return hints
