import re
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import List, Dict, Optional
from dataclasses import dataclass
from fuzzywuzzy import fuzz, process

@dataclass
class ExtractionResult:
    """Container for extraction results with confidence."""
    value: str
    confidence: float
    position: int
    context: str

class KeywordExtractor:
    """Rule-based keyword extraction for stroke reports based on CSV variables."""
    
    def __init__(self):
        self.patterns = self._initialize_patterns()
        self.category_confidence = {}
        self.medical_abbreviations = self._initialize_medical_abbreviations()
        self.multilingual_terms = self._initialize_multilingual_terms()
        
    def _initialize_patterns(self):
        # Define extraction patterns based on CSV variables
        patterns = {
            'anaesthesia': [
                r'\bitn\b',  # Intubationsnarkose abbreviation
                r'\bintubation\b',
                r'\bbetäubung\b',
                r'\banästhesie\b',  # More general form
                r'\blokal.*betäub\w*\b',  # Local anesthesia variations
                r'\bspinal.*anästhesie\b',  # Spinal anesthesia
                r'\bregional.*anästhesie\b',  # Regional anesthesia
                r'\bpropofol\b',  # Common anesthetic drug
                r'\bmidazolam\b',  # Common sedative
                r'\bfentanyl\b',  # Common anesthetic
                r'\bsedier\w*\b',  # Sedation variations
                r'\bvollnarkose\b',  # General anesthesia
                r'\btiva\b',  # Total intravenous anesthesia
                r'\blaryngeal.*mask\b',  # Laryngeal mask airway
                r'\bendotracheal\b',  # Endotracheal intubation
                r'\bsevofluran\b',  # Inhalational anesthetic
                r'\bdesfluran\b',  # Inhalational anesthetic
                r'\bretrograde.*intubation\b',  # Special intubation technique
                r'\bmonitored.*anesthesia\b',  # MAC - Monitored anesthesia care
            ],
            
            'aspiration_catheter_used': [
                r'\baspiration.*katheter\b',
                r'\baspiration.*device\b',
                r'\bmax\s+ace\b',  # Penumbra device
                r'\bmax\s+rep\b',  # Penumbra device
                r'\b3d\s+revascularization\b',  # Penumbra device
                r'\bsolumbra\b',  # Combined technique
                r'\badapt\b',  # Aspiration device
                r'\breact\b',  # Aspiration catheter
                r'\bpenumbra\b',  # Penumbra system
                r'\bsofia\b',  # SOFIA catheter
                r'\bjet\s+7\b',  # JET 7 catheter
                r'\bace\s+64\b',  # ACE 64 catheter
                r'\bace\s+68\b',  # ACE 68 catheter
                r'\bvac\s+lock\b',  # VAC Lock system
                r'\bsupport\b',  # Support catheter
                r'\bballoon.*guide\b',  # Balloon guide catheter
                r'\bsaugkatheter\b',  # German: suction catheter
                r'\bashpirations.*technik\b',  # Aspiration technique
            ],
            
            'guide_catheter_used': [
                r'\bguide.?katheter\b',
                r'\bführungskatheter\b'
            ],
            
            'microcatheter_used': [
                r'\bmikrokatheter\b',
                r'\bmicrocatheter\b'
            ],
            
            'stent_retriever_used': [
                r'\bstent.?retriever\b',
                r'\btrevo\b',
                r'\bsolitaire\b',
                r'\bembotrap\b',
                r'\bcatch\s+mini\b',  # Catch Mini device
                r'\bcatch\s+view\b',  # Catch View device
                r'\brevive\b',  # Revive device
                r'\beric\b',  # ERIC retriever
                r'\btigertriever\b',  # TigerTriever
                r'\bmindaloop\b',  # MindaLoop
                r'\bphenox\b',  # Phenox devices
                r'\bstentriever\b',  # Alternative spelling
                r'\bmechanisch.*rekanalisation\b',  # Mechanical recanalization
                r'\bstent.*basierte.*technik\b',  # Stent-based technique
            ],
            
            'tici_score': [
                r'\btici\s*[0-3][abc]?\b',
                r'\breperfusionsergebnis\b',
                r'\brekanalisierung\b'
            ],
            
            'periprocedural_ia_thrombolysis': [
                r'\bia.?thrombolyse\b',
                r'\bintra.?arterial.*thrombolyse\b',
                r'\brtpa\b',
                r'\balteplas[e]?\b',
                r'\btenecteplas[e]?\b',
                r'\burkinas[e]?\b'
            ],
            
            'periprocedural_antiplatelet': [
                r'\bthrombozytenaggregationshemmung\b',
                r'\baspirin\b',
                r'\bclopidogrel\b'
            ],
            
            'complications': [
                r'\bkompliziert\w*\b',
                r'\bembolie\b',
                r'\bthrombose\b',
                r'\bverschluss\b',
                r'\bokklusion\b',
                r'\breperfusion.*schaden\b',
                r'\bmaligne.*ödem\b',
                r'\bintrakranielle.*blutung\b',
                r'\bich\b',  # Intracerebral hemorrhage
                r'\bsah\b',  # Subarachnoid hemorrhage
                r'\bvasosp\w*\b',  # Vasospasm
                r'\bhydrozephal\w*\b',  # Hydrocephalus
            ],
            
            'site_of_occlusion': [
                r'\bgefäßverschlüsse\b',
                r'\bverschluss\b',
                r'\bokklusion\b',
                r'\bmca\b',
                r'\bica\b'
            ],
            
            'stenoses_cervical_arteries': [
                r'\bstenosen.*zervikalen\b',
                r'\bhalsarterien.*stenose\b'
            ],
            
            'extracranial_pta_stenting': [
                r'\bextrakranielle\s+pta\b',
                r'\bextrakranielle.*stenting\b'
            ],
            
            'intracranial_pta_stenting': [
                r'\bintrakranielle\s+pta\b',
                r'\bintrakranielle.*stenting\b'
            ],
            
            'technique_first_maneuver': [
                r'\btechnik.*manöver\b',
                r'\berste.*technik\b'
            ],
            
            'visualisation_vessels': [
                r'\bdarstellung.*gefäße\b',
                r'\bangiographie\b',
                r'\bdsa\b'
            ],
            
            'number_recanalization_attempts': [
                r'\banzahl.*manöver\b',
                r'\b\d+.*manöver\b'
            ],
            
            'periprocedural_spasmolytic': [
                r'\bspasmolyse\b',
                r'\bvasospasmen\b',
                r'\bnimodipin\b'
            ]
        }
        
        # Time patterns for start and end times
        self.start_time_patterns = [
            r'schleuse.*aufgegeben.*?([0-2]?[0-9]:[0-5][0-9])',
            r'beginn.*intervention.*?([0-2]?[0-9]:[0-5][0-9])',
            r'start.*?([0-2]?[0-9]:[0-5][0-9])',
            r'interventionsbeginn.*?([0-2]?[0-9]:[0-5][0-9])'
        ]
        
        self.end_time_patterns = [
            r'schleuse.*entfernt.*?([0-2]?[0-9]:[0-5][0-9])',
            r'ende.*intervention.*?([0-2]?[0-9]:[0-5][0-9])',
            r'abschluss.*?([0-2]?[0-9]:[0-5][0-9])'
        ]
        
        # General time patterns
        self.general_time_patterns = [
            r'\b([0-2]?[0-9]:[0-5][0-9])\b',  # HH:MM format
            r'uhrzeit:?\s*([0-2]?[0-9]:[0-5][0-9])'
        ]
        return patterns
    
    def _initialize_medical_abbreviations(self):
        """Initialize medical abbreviations with their full forms for fuzzy matching."""
        return {
            'rtpa': ['alteplase', 'tenecteplase', 'recombinant tissue plasminogen activator'],
            'tici': ['thrombolysis in cerebral infarction', 'recanalization score'],
            'ica': ['internal carotid artery', 'arteria carotis interna'],
            'mca': ['middle cerebral artery', 'arteria cerebri media'],
            'pca': ['posterior cerebral artery', 'arteria cerebri posterior'],
            'aca': ['anterior cerebral artery', 'arteria cerebri anterior'],
            'eva': ['external carotid artery', 'arteria carotis externa'],
            'dsa': ['digital subtraction angiography', 'digitale subtraktionsangiographie'],
            'cta': ['computed tomography angiography', 'computertomographie angiographie'],
            'mra': ['magnetic resonance angiography', 'magnetresonanzangiographie'],
            'ich': ['intracerebral hemorrhage', 'intrazerebrale blutung'],
            'sah': ['subarachnoid hemorrhage', 'subarachnoidalblutung'],
            'ivh': ['intraventricular hemorrhage', 'intraventrikuläre blutung'],
            'nihss': ['national institutes of health stroke scale'],
            'aspects': ['alberta stroke program early ct score'],
            'mrs': ['modified rankin scale', 'modifizierte rankin skala']
        }
    
    def _initialize_multilingual_terms(self):
        """Initialize multilingual medical terms (English/German)."""
        return {
            'anesthesia': ['anesthesia', 'anästhesie', 'narkose', 'sedation', 'sedierung'],
            'catheter': ['catheter', 'katheter', 'device', 'gerät'],
            'stent': ['stent', 'stentretriever', 'stent retriever'],
            'thrombosis': ['thrombosis', 'thrombose', 'clot', 'thrombus'],
            'hemorrhage': ['hemorrhage', 'blutung', 'bleeding', 'hematoma', 'hämatom'],
            'occlusion': ['occlusion', 'verschluss', 'blockage', 'stenosis', 'stenose'],
            'recanalization': ['recanalization', 'rekanalisation', 'reperfusion', 'reperfusion'],
            'intervention': ['intervention', 'procedure', 'prozedur', 'eingriff'],
            'complication': ['complication', 'komplikation', 'adverse event', 'nebenwirkung']
        }
    
    def extract_category(self, text: str, category: str) -> List[ExtractionResult]:
        """Extract keywords for a specific category with confidence scoring."""
        if category not in self.patterns:
            return []
        
        results = []
        
        # Rule-based extraction with confidence
        for pattern in self.patterns[category]:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                start = max(0, match.start() - 30)
                end = min(len(text), match.end() + 30)
                context = text[start:end].strip()
                
                # Calculate base confidence based on pattern specificity
                base_confidence = self._calculate_pattern_confidence(pattern, match.group())
                
                # Adjust confidence based on context
                final_confidence = self._adjust_confidence_by_context(
                    base_confidence, context, match.start(), category
                )
                
                result = ExtractionResult(
                    value=match.group(),
                    confidence=final_confidence,
                    position=match.start(),
                    context=context
                )
                results.append(result)
        
        # Add fuzzy matching results
        fuzzy_results = self._fuzzy_match_category(text, category)
        results.extend(fuzzy_results)
        
        return results
    
    def _calculate_pattern_confidence(self, pattern: str, match: str) -> float:
        """Calculate confidence based on pattern specificity."""
        # More specific patterns get higher confidence
        if len(pattern) > 20:  # Very specific patterns
            return 0.95
        elif len(pattern) > 10:  # Moderately specific
            return 0.85
        elif len(match) > 3:  # Longer matches are more reliable
            return 0.75
        else:
            return 0.65
    
    def _adjust_confidence_by_context(self, base_confidence: float, context: str, 
                                    position: int, category: str) -> float:
        """Adjust confidence based on surrounding context."""
        confidence = base_confidence
        
        # Check for negation
        if self.has_negation(context, len(context)//2):
            confidence *= 0.3  # Heavily reduce confidence for negated terms
        
        # Check for uncertainty indicators
        uncertainty_patterns = [r'\bmöglich\b', r'\bverdacht\b', r'\bwahrscheinlich\b', 
                              r'\bpossible\b', r'\bsuspected\b', r'\blikely\b']
        for pattern in uncertainty_patterns:
            if re.search(pattern, context, re.IGNORECASE):
                confidence *= 0.7
                break
        
        # Boost confidence for category-specific context
        if category == 'tici_score' and re.search(r'\bscore\b|\bergebnis\b', context, re.IGNORECASE):
            confidence *= 1.1
        elif category == 'anaesthesia' and re.search(r'\bpatient\b|\beinleitung\b', context, re.IGNORECASE):
            confidence *= 1.1
        
        return min(confidence, 1.0)
    
    def _fuzzy_match_category(self, text: str, category: str) -> List[ExtractionResult]:
        """Perform fuzzy matching for medical abbreviations and multilingual terms."""
        results = []
        
        # Split text into words for fuzzy matching
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Check abbreviations
        for word in words:
            if word in self.medical_abbreviations:
                # Find position in original text
                pattern = r'\b' + re.escape(word) + r'\b'
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    start = max(0, match.start() - 30)
                    end = min(len(text), match.end() + 30)
                    context = text[start:end].strip()
                    
                    result = ExtractionResult(
                        value=match.group(),
                        confidence=0.9,  # High confidence for known abbreviations
                        position=match.start(),
                        context=context
                    )
                    results.append(result)
        
        # Check multilingual terms with fuzzy matching
        for term_group in self.multilingual_terms.values():
            for term in term_group:
                # Use fuzzy matching to find similar terms
                fuzzy_matches = process.extract(term, words, limit=3, scorer=fuzz.ratio)
                for match_data in fuzzy_matches:
                    match, score = match_data[0], match_data[1]
                    if score >= 80:  # High similarity threshold
                        pattern = r'\b' + re.escape(match) + r'\b'
                        text_matches = re.finditer(pattern, text, re.IGNORECASE)
                        for text_match in text_matches:
                            start = max(0, text_match.start() - 30)
                            end = min(len(text), text_match.end() + 30)
                            context = text[start:end].strip()
                            
                            result = ExtractionResult(
                                value=text_match.group(),
                                confidence=score / 100.0,  # Convert to 0-1 scale
                                position=text_match.start(),
                                context=context
                            )
                            results.append(result)
        
        return results
    
    def extract_start_time(self, text: str) -> List[ExtractionResult]:
        """Extract start time of intervention."""
        results = []
        
        for pattern in self.start_time_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                time_value = match.group(1)
                
                start = max(0, match.start() - 20)
                end = min(len(text), match.end() + 20)
                context = text[start:end].strip()
                
                result = ExtractionResult(
                    value=time_value,
                    confidence=0.95,
                    position=match.start(),
                    context=context
                )
                results.append(result)
        
        return results
    
    def extract_end_time(self, text: str) -> List[ExtractionResult]:
        """Extract end time of intervention."""
        results = []
        
        for pattern in self.end_time_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                time_value = match.group(1)
                
                start = max(0, match.start() - 20)
                end = min(len(text), match.end() + 20)
                context = text[start:end].strip()
                
                result = ExtractionResult(
                    value=time_value,
                    confidence=0.95,
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
        """Extract all categories from text based on CSV variables with confidence scoring."""
        results = {
            'report_id': report_id or 'unknown',
            'text_length': len(text),
            'confidence_scores': {},
            'manual_review_priority': 'low'
        }
        
        low_confidence_count = 0
        total_extractions = 0
        
        # Extract each category
        for category in self.patterns.keys():
            extractions = self.extract_category(text, category)
            if extractions:
                # Sort by confidence and take the highest confidence result
                extractions.sort(key=lambda x: x.confidence, reverse=True)
                best_extraction = extractions[0]
                
                results[category] = best_extraction.value
                results['confidence_scores'][category] = best_extraction.confidence
                
                # Track low confidence extractions
                if best_extraction.confidence < 0.7:
                    low_confidence_count += 1
                total_extractions += 1
            else:
                results[category] = None
                results['confidence_scores'][category] = 0.0
        
        # Extract start and end times separately
        start_time_extractions = self.extract_start_time(text)
        if start_time_extractions:
            best_start_time = max(start_time_extractions, key=lambda x: x.confidence)
            results['start_time_intervention'] = best_start_time.value
            results['confidence_scores']['start_time_intervention'] = best_start_time.confidence
        else:
            results['start_time_intervention'] = None
            results['confidence_scores']['start_time_intervention'] = 0.0
        
        end_time_extractions = self.extract_end_time(text)
        if end_time_extractions:
            best_end_time = max(end_time_extractions, key=lambda x: x.confidence)
            results['end_time_intervention'] = best_end_time.value
            results['confidence_scores']['end_time_intervention'] = best_end_time.confidence
        else:
            results['end_time_intervention'] = None
            results['confidence_scores']['end_time_intervention'] = 0.0
        
        # Calculate manual review priority
        results['manual_review_priority'] = self._calculate_review_priority(
            low_confidence_count, total_extractions, results['confidence_scores']
        )
        
        return results
    
    def _calculate_review_priority(self, low_confidence_count: int, total_extractions: int, 
                                 confidence_scores: Dict) -> str:
        """Calculate priority for manual review based on extraction confidence."""
        if total_extractions == 0:
            return 'high'  # No extractions found - needs review
        
        # Calculate average confidence
        non_zero_scores = [score for score in confidence_scores.values() if score > 0]
        if not non_zero_scores:
            return 'high'
        
        avg_confidence = sum(non_zero_scores) / len(non_zero_scores)
        low_confidence_ratio = low_confidence_count / total_extractions
        
        # High priority: low average confidence or many low confidence extractions
        if avg_confidence < 0.6 or low_confidence_ratio > 0.5:
            return 'high'
        # Medium priority: moderate confidence issues
        elif avg_confidence < 0.8 or low_confidence_ratio > 0.3:
            return 'medium'
        else:
            return 'low'

def extract_csv_variables_rule_based(text: str, report_id: Optional[str] = None) -> Dict[str, Optional[str]]:
    """Extract CSV variables using rule-based approach."""
    extractor = KeywordExtractor()
    return extractor.extract_all(text, report_id)

if __name__ == "__main__":
    # Test the extractor
    test_text = """Patient mit akutem ischämischem Schlaganfall. Allgemeinanästhesie eingeleitet.
                   Interventionsbeginn: 08:30 Uhr. rtPA bereits präklinisch verabreicht.
                   Mechanische Thrombektomie mit Trevo Stentretriever durchgeführt.
                   TICI 3 Rekanalisierung erreicht. Keine intraoperativen Komplikationen."""
    
    extractor = KeywordExtractor()
    results = extractor.extract_all(test_text, "test_001")
    
    print("📊 Rule-based CSV Variable Extraction Test")
    print("=" * 50)
    
    for variable, value in results.items():
        if value is not None and variable not in ['report_id', 'text_length']:
            print(f"  ✅ {variable}: {value}")
        elif variable not in ['report_id', 'text_length']:
            print(f"  ❌ {variable}: Not found")