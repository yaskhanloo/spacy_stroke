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
    """Rule-based keyword extraction for stroke reports based on CSV variables."""
    
    def __init__(self):
        # Define extraction patterns based on CSV variables
        self.patterns = {
            'anaesthesia': [
                r'\bintubationsnarkose\b',
                r'\ballgemeinanästhesie\b',
                r'\bsedierung\b',
                r'\blokalanästhesie\b',
                r'\bnarkose\b',
                r'\bvollnarkose\b'
            ],
            
            'aspiration_catheter_used': [
                r'\baspirationskatheter\b',
                r'\bsofia\b',
                r'\bpenumbra\b',
                r'\bcatch\s+mini\b'
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
                r'\bembotrap\b'
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
                r'\bkomplikationen?\b',
                r'\bperforation\b',
                r'\bblutung\b',
                r'\bnachblutung\b',
                r'\bhämatom\b',
                r'\bischämie\b',
                r'\binfarkt\b',
                r'\bödem\b'
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
        """Extract all categories from text based on CSV variables."""
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
        
        # Extract start and end times separately
        start_time_extractions = self.extract_start_time(text)
        if start_time_extractions:
            unique_start_times = list(set([r.value for r in start_time_extractions]))
            results['start_time_intervention'] = unique_start_times[0] if unique_start_times else None
        else:
            results['start_time_intervention'] = None
        
        end_time_extractions = self.extract_end_time(text)
        if end_time_extractions:
            unique_end_times = list(set([r.value for r in end_time_extractions]))
            results['end_time_intervention'] = unique_end_times[0] if unique_end_times else None
        else:
            results['end_time_intervention'] = None
        
        return results

def extract_csv_variables_rule_based(text: str, report_id: str = None) -> Dict[str, Optional[str]]:
    """Extract CSV variables using rule-based approach."""
    extractor = KeywordExtractor()
    return extractor.extract_all(text, report_id)

if __name__ == "__main__":
    # Test the updated extractor
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