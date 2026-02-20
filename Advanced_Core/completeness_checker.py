import re
import numpy as np
from typing import Dict, List, Set, Tuple
from collections import Counter

class CompletenessChecker:
    """
    Checks if answer completely addresses the question
    Metrics: Keyword coverage, question type requirements, thoroughness
    """
    
    def __init__(self):
        # Question type patterns
        self.question_patterns = {
            'definition': {
                'keywords': ['define', 'what is', 'meaning of', 'explain the term'],
                'requirements': ['clear definition', 'key characteristics', 'examples'],
                'weight_concepts': 0.6,
                'weight_examples': 0.4
            },
            'explanation': {
                'keywords': ['explain', 'describe', 'discuss', 'elaborate'],
                'requirements': ['step-by-step', 'causes/effects', 'examples', 'details'],
                'weight_steps': 0.4,
                'weight_details': 0.6
            },
            'comparison': {
                'keywords': ['compare', 'contrast', 'difference', 'similarities'],
                'requirements': ['similarities', 'differences', 'both sides'],
                'weight_similarities': 0.4,
                'weight_differences': 0.6
            },
            'advantages': {
                'keywords': ['advantages', 'benefits', 'pros', 'strengths'],
                'requirements': ['multiple points', 'examples', 'evidence'],
                'weight_count': 0.5,
                'weight_details': 0.5
            },
            'disadvantages': {
                'keywords': ['disadvantages', 'drawbacks', 'cons', 'limitations'],
                'requirements': ['multiple points', 'examples', 'solutions'],
                'weight_count': 0.5,
                'weight_solutions': 0.5
            },
            'process': {
                'keywords': ['how', 'process', 'steps', 'procedure', 'mechanism'],
                'requirements': ['sequence', 'steps', 'order', 'stages'],
                'weight_sequence': 0.6,
                'weight_details': 0.4
            }
        }
    
    def check_completeness(self, question: str, student_answer: str, 
                          ideal_answer: str = None) -> Dict:
        """
        Check if answer completely addresses the question
        """
        # Classify question type
        question_type = self._classify_question(question)
        
        # Extract question keywords
        question_keywords = self._extract_keywords(question)
        student_keywords = self._extract_keywords(student_answer)
        
       