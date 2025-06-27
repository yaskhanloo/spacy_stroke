# NLP Debugging & Testing Framework
# Learn to systematically debug and improve your NLP system

import re
import json
from typing import List, Dict, Set, Tuple
from dataclasses import dataclass
from collections import defaultdict, Counter

@dataclass
class TestCase:
    """Structured test case for NLP evaluation."""
    text: str
    expected: Dict[str, List[str]]
    description: str
    difficulty: str = "easy"  # easy, medium, hard
    
@dataclass
class ExtractionResult:
    """Result of extraction with metadata."""
    category: str
    value: str
    confidence: float
    start_pos: int
    end_pos: int
    context: str

class NLPTester:
    """Framework for systematic NLP testing and debugging."""
    
    def __init__(self):
        self.test_cases = []
        self.results = []
        
    def add_test_case(self, text: str, expected: Dict[str, List[str]], 
                     description: str, difficulty: str = "easy"):
        """Add a test case to the test suite."""
        self.test_cases.append(TestCase(text, expected, description, difficulty))
    
    def create_medical_test_suite(self):
        """Create comprehensive test cases for medical text extraction."""
        
        # Basic cases - should work perfectly
        self.add_test_case(
            text="Patient in Allgemeinanästhesie, rtPA gegeben",
            expected={
                'anesthesia': ['allgemeinanästhesie'],
                'medication': ['rtpa']
            },
            description="Basic extraction - clear terms",
            difficulty="easy"
        )
        
        # Case variations
        self.add_test_case(
            text="PATIENT UNTER VOLLNARKOSE, HEPARIN VERABREICHT",
            expected={
                'anesthesia': ['vollnarkose'],
                'medication': ['heparin']
            },
            description="All caps text",
            difficulty="easy"
        )
        
        # Multiple items
        self.add_test_case(
            text="Sedierung und rtPA, zusätzlich Heparin und Aspirin",
            expected={
                'anesthesia': ['sedierung'],
                'medication': ['rtpa', 'heparin', 'aspirin']
            },
            description="Multiple medications",
            difficulty="medium"
        )
        
        # Edge cases - tricky scenarios
        self.add_test_case(
            text="Patient Sofia wurde mit SOFIA Katheter behandelt",
            expected={
                'device': ['sofia'],  # Should extract device, not patient name
                'anesthesia': [],
                'medication': []
            },
            description="Ambiguous term (Sofia = name vs device)",
            difficulty="hard"
        )
        
        # Negations
        self.add_test_case(
            text="Keine rtPA Gabe, keine Vollnarkose",
            expected={
                'anesthesia': [],
                'medication': []
            },
            description="Negated terms - should not extract",
            difficulty="hard"
        )
        
        # Typos and variations
        self.add_test_case(
            text="Allgemein-Anästhesie, rt-PA verabreicht",
            expected={
                'anesthesia': ['allgemein-anästhesie'],
                'medication': ['rt-pa']
            },
            description="Hyphenated variations",
            difficulty="medium"
        )
        
        # Context dependent
        self.add_test_case(
            text="Trevo Stentretriever im Aneurysma-Sack, nicht bei Schlaganfall",
            expected={
                'device': ['trevo']
            },
            description="Context matters for relevance",
            difficulty="hard"
        )

class DebugExtractor:
    """Enhanced extractor with debugging capabilities."""
    
    def __init__(self):
        # Basic patterns with metadata
        self.patterns = {
            'anesthesia': {
                'patterns': [
                    (r'\ballgemeinanästhesie\b', 'general anesthesia'),
                    (r'\ballgemein-anästhesie\b', 'general anesthesia (hyphenated)'),
                    (r'\bvollnarkose\b', 'general anesthesia'),
                    (r'\bsedierung\b', 'sedation'),
                    (r'\blokalanästhesie\b', 'local anesthesia'),
                ],
                'confidence': 0.95
            },
            'medication': {
                'patterns': [
                    (r'\brtpa\b', 'tissue plasminogen activator'),
                    (r'\brt-pa\b', 'tissue plasminogen activator (hyphenated)'),
                    (r'\bheparin\b', 'anticoagulant'),
                    (r'\baspirin\b', 'antiplatelet'),
                    (r'\burkinas[e]?\b', 'fibrinolytic'),
                ],
                'confidence': 0.9
            },
            'device': {
                'patterns': [
                    (r'\bsofia\b', 'aspiration catheter'),
                    (r'\btrevo\b', 'stent retriever'),
                    (r'\bsolitaire\b', 'stent retriever'),
                    (r'\bcatch\s+mini\b', 'stent retriever'),
                ],
                'confidence': 0.85
            }
        }
        
        # Negative patterns (should NOT extract if found)
        self.negative_patterns = [
            r'\bkein[e]?\s+',     # "keine rtPA" 
            r'\bnicht\s+',        # "nicht rtPA"
            r'\bohne\s+',         # "ohne rtPA"
        ]
        
        # Context patterns (increase/decrease confidence)
        self.positive_context = [
            r'\bgegeben\b', r'\bverabreicht\b', r'\bangewendet\b'
        ]
        self.negative_context = [
            r'\babgelehnt\b', r'\bverweigert\b', r'\bkontraindiziert\b'
        ]
    
    def debug_extract(self, text: str, category: str) -> List[ExtractionResult]:
        """Extract with detailed debugging information."""
        results = []
        
        if category not in self.patterns:
            return results
        
        text_lower = text.lower()
        
        # Check each pattern
        for pattern, description in self.patterns[category]['patterns']:
            matches = re.finditer(pattern, text_lower, re.IGNORECASE)
            
            for match in matches:
                start, end = match.span()
                value = match.group()
                
                # Get context (20 chars before and after)
                context_start = max(0, start - 20)
                context_end = min(len(text), end + 20)
                context = text[context_start:context_end]
                
                # Calculate confidence based on context
                confidence = self.patterns[category]['confidence']
                
                # Check for negation
                negation_found = False
                for neg_pattern in self.negative_patterns:
                    if re.search(neg_pattern + r'.*?' + re.escape(value), text_lower):
                        negation_found = True
                        confidence *= 0.1  # Heavily penalize negated terms
                        break
                
                # Adjust confidence based on context
                for pos_pattern in self.positive_context:
                    if re.search(pos_pattern, context.lower()):
                        confidence = min(1.0, confidence * 1.2)
                        break
                
                for neg_pattern in self.negative_context:
                    if re.search(neg_pattern, context.lower()):
                        confidence *= 0.3
                        break
                
                result = ExtractionResult(
                    category=category,
                    value=value,
                    confidence=confidence,
                    start_pos=start,
                    end_pos=end,
                    context=context.strip()
                )
                
                results.append(result)
        
        return results
    
    def extract_all_debug(self, text: str) -> Dict[str, List[ExtractionResult]]:
        """Extract all categories with debugging info."""
        results = {}
        for category in self.patterns.keys():
            results[category] = self.debug_extract(text, category)
        return results

class NLPAnalyzer:
    """Analyze and improve NLP system performance."""
    
    def __init__(self, extractor: DebugExtractor):
        self.extractor = extractor
        self.error_analysis = defaultdict(list)
    
    def run_test_suite(self, tester: NLPTester) -> Dict:
        """Run all test cases and analyze results."""
        results = {
            'total_tests': len(tester.test_cases),
            'passed': 0,
            'failed': 0,
            'categories': defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0}),
            'failure_analysis': []
        }
        
        for i, test_case in enumerate(tester.test_cases):
            print(f"\n--- Test {i+1}: {test_case.description} ({test_case.difficulty}) ---")
            print(f"Text: '{test_case.text}'")
            
            # Extract with our system
            extracted = self.extractor.extract_all_debug(test_case.text)
            
            # Convert to simple format for comparison
            extracted_simple = {}
            for category, extractions in extracted.items():
                extracted_simple[category] = [e.value for e in extractions if e.confidence > 0.5]
            
            # Compare with expected
            test_passed = True
            for category in test_case.expected:
                expected_set = set(test_case.expected[category])
                extracted_set = set(extracted_simple.get(category, []))
                
                # Calculate metrics for this category
                tp = len(expected_set & extracted_set)
                fp = len(extracted_set - expected_set)
                fn = len(expected_set - extracted_set)
                
                results['categories'][category]['tp'] += tp
                results['categories'][category]['fp'] += fp
                results['categories'][category]['fn'] += fn
                
                print(f"  {category.upper()}:")
                print(f"    Expected: {sorted(list(expected_set))}")
                print(f"    Extracted: {sorted(list(extracted_set))}")
                print(f"    TP={tp}, FP={fp}, FN={fn}")
                
                if expected_set != extracted_set:
                    test_passed = False
                    
                    # Detailed error analysis
                    if fp > 0:
                        false_positives = extracted_set - expected_set
                        for fp_item in false_positives:
                            # Find the extraction details
                            for extraction in extracted[category]:
                                if extraction.value == fp_item:
                                    self.error_analysis['false_positives'].append({
                                        'text': test_case.text,
                                        'category': category,
                                        'extracted': fp_item,
                                        'confidence': extraction.confidence,
                                        'context': extraction.context,
                                        'reason': 'Pattern matched but should not have'
                                    })
                    
                    if fn > 0:
                        false_negatives = expected_set - extracted_set
                        for fn_item in false_negatives:
                            self.error_analysis['false_negatives'].append({
                                'text': test_case.text,
                                'category': category,
                                'missed': fn_item,
                                'reason': 'Expected term not found by any pattern'
                            })
            
            if test_passed:
                results['passed'] += 1
                print("  ✅ PASSED")
            else:
                results['failed'] += 1
                print("  ❌ FAILED")
                results['failure_analysis'].append({
                    'test_case': test_case,
                    'extracted': extracted_simple
                })
        
        return results
    
    def generate_improvement_report(self, results: Dict):
        """Generate recommendations for system improvement."""
        print("\n" + "="*60)
        print("IMPROVEMENT ANALYSIS REPORT")
        print("="*60)
        
        # Overall metrics
        print(f"\nOVERALL PERFORMANCE:")
        print(f"Tests passed: {results['passed']}/{results['total_tests']} ({results['passed']/results['total_tests']*100:.1f}%)")
        
        # Category-wise metrics
        print(f"\nCATEGORY PERFORMANCE:")
        for category, metrics in results['categories'].items():
            tp, fp, fn = metrics['tp'], metrics['fp'], metrics['fn']
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            print(f"  {category.upper()}:")
            print(f"    Precision: {precision:.3f}")
            print(f"    Recall: {recall:.3f}")
            print(f"    F1-Score: {f1:.3f}")
        
        # Error analysis
        print(f"\nERROR ANALYSIS:")
        
        if self.error_analysis['false_positives']:
            print(f"\n❌ FALSE POSITIVES ({len(self.error_analysis['false_positives'])}):")
            fp_by_category = defaultdict(list)
            for fp in self.error_analysis['false_positives']:
                fp_by_category[fp['category']].append(fp)
            
            for category, fps in fp_by_category.items():
                print(f"  {category.upper()}:")
                for fp in fps[:3]:  # Show first 3
                    print(f"    • '{fp['extracted']}' in context: '{fp['context']}'")
                    print(f"      Confidence: {fp['confidence']:.2f}")
        
        if self.error_analysis['false_negatives']:
            print(f"\n❌ FALSE NEGATIVES ({len(self.error_analysis['false_negatives'])}):")
            fn_by_category = defaultdict(list)
            for fn in self.error_analysis['false_negatives']:
                fn_by_category[fn['category']].append(fn)
            
            for category, fns in fn_by_category.items():
                print(f"  {category.upper()}:")
                for fn in fns[:3]:  # Show first 3
                    print(f"    • Missed '{fn['missed']}' in: '{fn['text']}'")
        
        # Recommendations
        print(f"\n💡 IMPROVEMENT RECOMMENDATIONS:")
        
        # Pattern suggestions based on false negatives
        missing_terms = defaultdict(set)
        for fn in self.error_analysis['false_negatives']:
            missing_terms[fn['category']].add(fn['missed'])
        
        for category, terms in missing_terms.items():
            if terms:
                print(f"\n  {category.upper()}:")
                print(f"    Add patterns for: {', '.join(terms)}")
                
                # Suggest specific pattern improvements
                if category == 'medication':
                    for term in terms:
                        if '-' in term:
                            print(f"    Consider hyphenated pattern: r'\\b{term.replace('-', '-?')}\\b'")
                        else:
                            print(f"    Add pattern: r'\\b{term}\\b'")
        
        # Context analysis
        print(f"\n  CONTEXT IMPROVEMENTS:")
        print(f"    • Implement negation detection for 'keine', 'nicht', 'ohne'")
        print(f"    • Add context scoring for medical vs non-medical usage")
        print(f"    • Consider abbreviation expansion (GA -> General Anesthesia)")
        
        # Technical recommendations
        print(f"\n  TECHNICAL IMPROVEMENTS:")
        print(f"    • Add fuzzy matching for typos")
        print(f"    • Implement confidence thresholding")
        print(f"    • Add term disambiguation (Sofia name vs device)")
        print(f"    • Consider machine learning for complex cases")

def main_debug_session():
    """Main function to run a complete debugging session."""
    print("🔍 NLP DEBUGGING SESSION")
    print("="*50)
    
    # Initialize components
    tester = NLPTester()
    tester.create_medical_test_suite()
    
    extractor = DebugExtractor()
    analyzer = NLPAnalyzer(extractor)
    
    # Run tests
    results = analyzer.run_test_suite(tester)
    
    # Generate improvement report
    analyzer.generate_improvement_report(results)
    
    print(f"\n🎯 NEXT STEPS:")
    print(f"1. Fix patterns based on false negatives")
    print(f"2. Add negation detection")
    print(f"3. Implement confidence thresholding")
    print(f"4. Re-run tests to measure improvement")
    print(f"5. Add more challenging test cases")

if __name__ == "__main__":
    main_debug_session()