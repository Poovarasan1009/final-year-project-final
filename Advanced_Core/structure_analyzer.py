import re
import numpy as np
from typing import Dict, List, Tuple
from collections import Counter

class StructureAnalyzer:
    """
    Analyzes answer structure and coherence
    Metrics: Sentence count, connector words, paragraph structure, grammar
    """
    
    def __init__(self):
        # Connector words for coherence
        self.connectors = {
            'contrast': ['however', 'but', 'although', 'though', 'nevertheless',
                        'nonetheless', 'conversely', 'on the other hand',
                        'whereas', 'while', 'yet'],
            
            'addition': ['furthermore', 'moreover', 'in addition', 'additionally',
                        'also', 'besides', 'similarly', 'likewise'],
            
            'cause_effect': ['therefore', 'thus', 'consequently', 'hence', 'as a result',
                            'so', 'because', 'since', 'due to', 'owing to'],
            
            'sequence': ['first', 'second', 'third', 'next', 'then', 'after',
                        'before', 'finally', 'lastly', 'subsequently'],
            
            'example': ['for example', 'for instance', 'such as', 'namely',
                       'specifically', 'in particular'],
            
            'conclusion': ['in conclusion', 'to conclude', 'to sum up',
                          'in summary', 'overall', 'in brief']
        }
        
        # Academic style indicators
        self.academic_phrases = [
            'it can be argued that', 'this suggests that', 'the evidence shows',
            'according to', 'research indicates', 'studies have shown',
            'from this perspective', 'in terms of', 'with respect to'
        ]
    
    def analyze(self, text: str) -> Dict:
        """
        Comprehensive structure analysis
        Returns scores for various structural aspects
        """
        if not text or len(text.strip()) == 0:
            return self._empty_response()
        
        # Basic metrics
        sentence_count = self._count_sentences(text)
        word_count = self._count_words(text)
        paragraph_count = self._count_paragraphs(text)
        
        # Calculate individual scores
        sentence_score = self._sentence_structure_score(text, sentence_count)
        connector_score = self._connector_score(text)
        paragraph_score = self._paragraph_structure_score(text, paragraph_count)
        grammar_score = self._grammatical_coherence_score(text)
        academic_score = self._academic_style_score(text)
        
        # Weighted final score
        weights = {
            'sentence_structure': 0.25,
            'connectors': 0.25,
            'paragraph_structure': 0.20,
            'grammar': 0.20,
            'academic_style': 0.10
        }
        
        final_score = (
            weights['sentence_structure'] * sentence_score +
            weights['connectors'] * connector_score +
            weights['paragraph_structure'] * paragraph_score +
            weights['grammar'] * grammar_score +
            weights['academic_style'] * academic_score
        )
        
        return {
            'final_score': final_score,
            'sentence_count': sentence_count,
            'word_count': word_count,
            'paragraph_count': paragraph_count,
            'detailed_scores': {
                'sentence_structure': sentence_score,
                'connectors': connector_score,
                'paragraph_structure': paragraph_score,
                'grammar': grammar_score,
                'academic_style': academic_score
            },
            'connector_analysis': self._analyze_connectors(text),
            'feedback': self._generate_feedback({
                'sentence_structure': sentence_score,
                'connectors': connector_score,
                'paragraph_structure': paragraph_score,
                'grammar': grammar_score
            })
        }
    
    def _count_sentences(self, text: str) -> int:
        """Count sentences using punctuation and capitalization"""
        # Split by sentence-ending punctuation
        sentences = re.split(r'[.!?]+', text)
        # Filter out empty strings
        sentences = [s.strip() for s in sentences if s.strip()]
        return len(sentences)
    
    def _count_words(self, text: str) -> int:
        """Count words"""
        words = re.findall(r'\b\w+\b', text)
        return len(words)
    
    def _count_paragraphs(self, text: str) -> int:
        """Count paragraphs"""
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        return len(paragraphs)
    
    def _sentence_structure_score(self, text: str, sentence_count: int) -> float:
        """
        Score sentence structure
        Ideal: 3-8 sentences for short answer, varied lengths
        """
        if sentence_count == 0:
            return 0.0
        
        # Score based on sentence count
        if 3 <= sentence_count <= 8:
            count_score = 1.0
        elif sentence_count < 3:
            count_score = sentence_count / 3
        else:
            count_score = 8 / sentence_count
        
        # Analyze sentence length variation
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) < 2:
            variation_score = 0.5
        else:
            lengths = [len(re.findall(r'\b\w+\b', s)) for s in sentences]
            avg_length = np.mean(lengths)
            std_length = np.std(lengths)
            
            # Some variation is good, but not extreme
            if std_length > 0 and avg_length > 0:
                cv = std_length / avg_length  # Coefficient of variation
                if 0.3 <= cv <= 0.7:
                    variation_score = 1.0
                elif cv < 0.3:
                    variation_score = 0.7
                else:
                    variation_score = 0.5
            else:
                variation_score = 0.5
        
        return 0.6 * count_score + 0.4 * variation_score
    
    def _connector_score(self, text: str) -> float:
        """
        Score use of connector words
        More connectors = better flow, but don't overuse
        """
        text_lower = text.lower()
        
        # Count connectors by category
        connector_counts = {}
        total_connectors = 0
        
        for category, connectors in self.connectors.items():
            count = sum(1 for connector in connectors if connector in text_lower)
            connector_counts[category] = count
            total_connectors += count
        
        # Ideal: 2-5 connectors for short answer
        if total_connectors == 0:
            return 0.3
        elif 2 <= total_connectors <= 5:
            count_score = 1.0
        elif total_connectors < 2:
            count_score = total_connectors / 2
        else:
            count_score = 5 / total_connectors
        
        # Diversity score: use of different categories
        categories_used = sum(1 for count in connector_counts.values() if count > 0)
        diversity_score = categories_used / len(self.connectors)
        
        return 0.7 * count_score + 0.3 * diversity_score
    
    def _paragraph_structure_score(self, text: str, paragraph_count: int) -> float:
        """
        Score paragraph structure
        Ideal: 1-3 paragraphs for short answer
        """
        if paragraph_count == 0:
            return 0.0
        
        # Score based on paragraph count
        if 1 <= paragraph_count <= 3:
            count_score = 1.0
        elif paragraph_count > 3:
            count_score = 3 / paragraph_count
        else:
            count_score = paragraph_count  # Should be 1
        
        # Check paragraph lengths
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        if len(paragraphs) < 2:
            balance_score = 0.7
        else:
            lengths = [len(re.findall(r'\b\w+\b', p)) for p in paragraphs]
            avg_length = np.mean(lengths)
            # Check if paragraphs are reasonably balanced
            if max(lengths) / min(lengths) < 3:
                balance_score = 1.0
            else:
                balance_score = 0.5
        
        return 0.6 * count_score + 0.4 * balance_score
    
    def _grammatical_coherence_score(self, text: str) -> float:
        """
        Basic grammatical coherence check
        Simplified - in production use spaCy or NLTK
        """
        # Simple heuristics
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return 0.3
        
        # Check sentence starting with capital letter
        capital_start = 0
        for sentence in sentences:
            if sentence and sentence[0].isupper():
                capital_start += 1
        
        capital_score = capital_start / len(sentences)
        
        # Check sentence ending with punctuation
        proper_end = 0
        for sentence in text.split('\n'):
            sentence = sentence.strip()
            if sentence and sentence[-1] in '.!?':
                proper_end += 1
        
        if len(text.split('\n')) > 0:
            end_score = proper_end / len(text.split('\n'))
        else:
            end_score = 0.5
        
        # Check for very long sentences (potential run-ons)
        long_sentence_penalty = 0
        for sentence in sentences:
            words = re.findall(r'\b\w+\b', sentence)
            if len(words) > 40:  # Very long sentence
                long_sentence_penalty += 0.2
        
        penalty = min(long_sentence_penalty, 0.5)
        
        return (0.4 * capital_score + 0.4 * end_score) * (1 - penalty)
    
    def _academic_style_score(self, text: str) -> float:
        """Score academic writing style"""
        text_lower = text.lower()
        
        # Check for academic phrases
        academic_count = 0
        for phrase in self.academic_phrases:
            if phrase in text_lower:
                academic_count += 1
        
        # Check for informal language
        informal_words = ['really', 'very', 'a lot', 'stuff', 'thing', 'got', 'get']
        informal_count = sum(1 for word in informal_words if f' {word} ' in f' {text_lower} ')
        
        # Calculate score
        academic_score = min(academic_count / 3, 1.0)
        informal_penalty = min(informal_count / 3, 0.5)
        
        return academic_score * (1 - informal_penalty)
    
    def _analyze_connectors(self, text: str) -> Dict:
        """Detailed connector analysis"""
        text_lower = text.lower()
        analysis = {}
        
        for category, connectors in self.connectors.items():
            found = [connector for connector in connectors if connector in text_lower]
            analysis[category] = {
                'found': found,
                'count': len(found)
            }
        
        return analysis
    
    def _generate_feedback(self, scores: Dict) -> str:
        """Generate constructive feedback"""
        feedback = []
        
        if scores['sentence_structure'] < 0.5:
            feedback.append("Improve sentence structure. Aim for 3-8 sentences with varied lengths.")
        
        if scores['connectors'] < 0.4:
            feedback.append("Use more connecting words (however, therefore, for example) to improve flow.")
        
        if scores['paragraph_structure'] < 0.5:
            feedback.append("Organize answer into clear paragraphs (1-3 paragraphs for short answers).")
        
        if scores['grammar'] < 0.6:
            feedback.append("Check grammar: sentences should start with capital letters and end with proper punctuation.")
        
        if not feedback:
            feedback.append("Good structural organization. Clear and coherent answer.")
        
        return " | ".join(feedback)
    
    def _empty_response(self):
        """Return empty structure response"""
        return {
            'final_score': 0.0,
            'sentence_count': 0,
            'word_count': 0,
            'paragraph_count': 0,
            'detailed_scores': {
                'sentence_structure': 0.0,
                'connectors': 0.0,
                'paragraph_structure': 0.0,
                'grammar': 0.0,
                'academic_style': 0.0
            },
            'connector_analysis': {},
            'feedback': "No text provided for structure analysis."
        }