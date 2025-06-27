import re
from typing import List, Dict, Optional
from dataclasses import dataclass

@dataclass
class ExtractionResult:
    """Container for extraction results with confidence."""
    value: str
    confidence: float
    position: int
    context: str

class KeywordExtractor:
    """Rule-based keyword extraction for stroke reports."""
    
    def __init__(self):
        # Define extraction patterns
        self.patterns = {
            'anesthesia': [
                r'\ballgemeinanästhesie\b',
                r'\bsedierung\b',
                r'\blokalanästhesie\b',
                r'\bnarkose\b',
                r'\bvollnarkose\b'
            ],
            
            'medication': [
                r'\brtpa\b',
                r'\burkinas[e]?\b',
                r'\btenecteplas[e]?\b',
                r'\balteplas[e]?\b',
                r'\bheparin\b',
                r'\baspirin\b'
            ],
            
            'treatment_method': [
                r'\baspiration\b',
                r'\bstentretriever\b',
                r'\bthrombektomie\b',
                r'\bembolektomie\b',
                r'\bmechanische\s+rekanalisation\b',
                r'\bthrombolyse\b'
            ],
            
            'device': [
                r'\bsofia\b',
                r'\btrevo\b',
                r'\bcatch\s+mini\b',
                r'\bembotrap\b',
                r'\bsolitaire\b',
                r'\bpenumbra\b'
            ],
            
            'tici_score': [
                r'\btici\s*[0-3][abc]?\b',
                r'\btici\s*score\s*[0-3][abc]?\b'
            ],
            
            'complications': [
                r'\bperforation\b',
                r'\bblutung\b',
                r'\bhämatom\b',
                r'\bischämie\b',
                r'\binfarkt\b',
                r'\bödem\b',
                r'\bkomplikation\b'
            ]
        }
        
        # Time patterns (more specific)
        self.time_patterns = [
            r'\b([0-2]?[0-9]:[0-5][0-9])\b',  # HH:MM format
            r'beginn:?\s*([0-2]?[0-9]:[0-5][0-9])',
            r'start:?\s*([0-2]?[0-9]:[0-5][0-9])',
            r'uhrzeit:?\s*([0-2]?[0-9]:[0-5][0-9])'
        ]
    
    def extract_category(self, text: str, category: str) -> List[ExtractionResult]:
        """Extract keywords for a specific category."""
        results = []
        patterns = self.patterns.get(category, [])
        
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                # Get context (20 chars before and after)
                start = max(0, match.start() - 20)
                end = min(len(text), match.end() + 20)
                context = text[start:end].strip()
                
                result = ExtractionResult(
                    value=match.group(),
                    confidence=1.0,  # High confidence for exact matches
                    position=match.start(),
                    context=context
                )
                results.append(result)
        
        return results
    
    def extract_times(self, text: str) -> List[ExtractionResult]:
        """Extract time expressions."""
        results = []
        
        for pattern in self.time_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                # Extract just the time part
                time_value = match.group(1) if match.groups() else match.group()
                
                start = max(0, match.start() - 20)
                end = min(len(text), match.end() + 20)
                context = text[start:end].strip()
                
                result = ExtractionResult(
                    value=time_value,
                    confidence=0.9,
                    position=match.start(),
                    context=context
                )
                results.append(result)
        
        return results
    
    def has_negation(self, text: str, term_position: int) -> bool:
        """Check if a term is negated (keine, nicht, ohne)."""
        context = text[max(0, term_position-50):term_position]
        negation_patterns = [r'\bkein[e]?\s+', r'\bnicht\s+', r'\bohne\s+']
        for pattern in negation_patterns:
            if re.search(pattern, context, re.IGNORECASE):
                return True
        return False

    def extract_all(self, text: str, report_id: Optional[str] = None) -> Dict:
        """Extract all categories from text."""
        results = {
            'report_id': report_id or 'unknown',
            'text_length': len(text)
        }
        
        # Extract each category
        for category in self.patterns.keys():
            extractions = self.extract_category(text, category)
            if extractions:
                # Take unique values only
                unique_values = list(set([r.value for r in extractions]))
                results[category] = unique_values if len(unique_values) > 1 else unique_values[0]
            else:
                results[category] = None
        
        # Extract times separately
        time_extractions = self.extract_times(text)
        if time_extractions:
            unique_times = list(set([r.value for r in time_extractions]))
            results['times'] = unique_times if len(unique_times) > 1 else unique_times[0]
        else:
            results['times'] = None
        
        return results