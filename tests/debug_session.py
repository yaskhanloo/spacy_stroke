# Stroke Report Project - Live Debugging Session
# Let's debug your actual code systematically

import re
import sys
from typing import List, Dict, Set, Tuple
from dataclasses import dataclass
from collections import defaultdict

# First, let's recreate your current KeywordExtractor to debug it
class CurrentKeywordExtractor:
    """Your current extractor - let's see what's happening under the hood."""
    
    def __init__(self):
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
            ]
        }
    
    def extract_category(self, text: str, category: str) -> List[str]:
        """Extract keywords for a specific category - your current logic."""
        results = []
        patterns = self.patterns.get(category, [])
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            results.extend(matches)
        
        return list(set(results))  # Remove duplicates

# Enhanced debugging version
@dataclass
class DebugMatch:
    """Detailed information about a pattern match."""
    pattern: str
    matched_text: str
    start_pos: int
    end_pos: int
    context_before: str
    context_after: str
    confidence: float
    issues: List[str]

class DebugKeywordExtractor:
    """Enhanced version with detailed debugging info."""
    
    def __init__(self):
        # Same patterns as current version
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
            ]
        }
        
        # Problematic patterns we should watch for
        self.warning_patterns = {
            'negation': [r'\bkein[e]?\s+', r'\bnicht\s+', r'\bohne\s+'],
            'uncertainty': [r'\bmöglich\b', r'\bverdacht\b', r'\beventuell\b'],
            'past_tense': [r'\bwar\b', r'\bwurde\s+nicht\b'],
        }
    
    def debug_extract_category(self, text: str, category: str) -> List[DebugMatch]:
        """Extract with full debugging information."""
        results = []
        patterns = self.patterns.get(category, [])
        
        print(f"\n🔍 DEBUGGING {category.upper()} EXTRACTION")
        print(f"Text: '{text}'")
        print(f"Patterns to test: {len(patterns)}")
        
        for i, pattern in enumerate(patterns):
            print(f"\n  Pattern {i+1}: {pattern}")
            
            # Find all matches with position info
            matches = list(re.finditer(pattern, text, re.IGNORECASE))
            
            if not matches:
                print(f"    ❌ No matches found")
                continue
            
            for match in matches:
                start, end = match.span()
                matched_text = match.group()
                
                # Get context (30 chars before and after)
                context_start = max(0, start - 30)
                context_end = min(len(text), end + 30)
                
                context_before = text[context_start:start]
                context_after = text[end:context_end]
                full_context = text[context_start:context_end]
                
                # Analyze potential issues
                issues = []
                confidence = 1.0
                
                # Check for negation
                for neg_pattern in self.warning_patterns['negation']:
                    if re.search(neg_pattern + r'.*?' + re.escape(matched_text), 
                               text[max(0, start-50):end], re.IGNORECASE):
                        issues.append("NEGATION_DETECTED")
                        confidence *= 0.1
                
                # Check for uncertainty
                for unc_pattern in self.warning_patterns['uncertainty']:
                    if re.search(unc_pattern, full_context, re.IGNORECASE):
                        issues.append("UNCERTAINTY_CONTEXT")
                        confidence *= 0.7
                
                # Check for ambiguous terms (like Sofia)
                if category == 'device' and matched_text.lower() == 'sofia':
                    # Look for clues it might be a person name
                    if re.search(r'patient\s+sofia|frau\s+sofia|sofia\s+\w+\s+(jahre|alt)', 
                               full_context, re.IGNORECASE):
                        issues.append("POSSIBLE_PERSON_NAME")
                        confidence *= 0.3
                
                debug_match = DebugMatch(
                    pattern=pattern,
                    matched_text=matched_text,
                    start_pos=start,
                    end_pos=end, 
                    context_before=context_before,
                    context_after=context_after,
                    confidence=confidence,
                    issues=issues
                )
                
                results.append(debug_match)
                
                # Print detailed match info
                print(f"    ✅ MATCH: '{matched_text}' at position {start}-{end}")
                print(f"       Context: '...{context_before}[{matched_text}]{context_after}...'")
                print(f"       Confidence: {confidence:.2f}")
                if issues:
                    print(f"       ⚠️  Issues: {', '.join(issues)}")
        
        return results

def test_current_vs_debug_extractor():
    """Compare your current extractor with the debug version."""
    
    # Your sample reports
    test_reports = [
        {
            'id': 'report_001',
            'text': "Patient wurde in Allgemeinanästhesie behandelt. rtPA wurde verabreicht. TICI 3 erreicht."
        },
        {
            'id': 'problematic_01', 
            'text': "Patient Sofia wurde NICHT mit SOFIA Katheter behandelt. Keine rtPA Gabe."
        },
        {
            'id': 'problematic_02',
            'text': "Mögliche Sedierung geplant. rt-PA eventuell kontraindiziert."
        },
        {
            'id': 'problematic_03',
            'text': "Trevo war nicht verfügbar. Alternative: Solitaire Stentretriever."
        }
    ]
    
    current_extractor = CurrentKeywordExtractor()
    debug_extractor = DebugKeywordExtractor()
    
    print("="*80)
    print("STROKE REPORT DEBUGGING SESSION")
    print("="*80)
    
    for report in test_reports:
        print(f"\n{'='*60}")
        print(f"TESTING REPORT: {report['id']}")
        print(f"{'='*60}")
        print(f"Text: {report['text']}")
        
        # Test each category
        categories = ['anesthesia', 'medication', 'device', 'tici_score']
        
        for category in categories:
            print(f"\n--- {category.upper()} EXTRACTION ---")
            
            # Current extractor results
            current_results = current_extractor.extract_category(report['text'], category)
            print(f"Current extractor found: {current_results}")
            
            # Debug extractor results
            debug_results = debug_extractor.debug_extract_category(report['text'], category)
            
            # Summary
            high_confidence = [r for r in debug_results if r.confidence > 0.8]
            medium_confidence = [r for r in debug_results if 0.5 < r.confidence <= 0.8]
            low_confidence = [r for r in debug_results if r.confidence <= 0.5]
            
            print(f"\n  📊 CONFIDENCE BREAKDOWN:")
            print(f"    High confidence (>0.8): {[r.matched_text for r in high_confidence]}")
            print(f"    Medium confidence (0.5-0.8): {[r.matched_text for r in medium_confidence]}")
            print(f"    Low confidence (≤0.5): {[r.matched_text for r in low_confidence]}")
            
            # Issues summary
            all_issues = []
            for result in debug_results:
                all_issues.extend(result.issues)
            
            if all_issues:
                issue_counts = {}
                for issue in all_issues:
                    issue_counts[issue] = issue_counts.get(issue, 0) + 1
                print(f"    🚨 Issues found: {issue_counts}")

def analyze_pattern_weaknesses():
    """Systematically find weaknesses in your current patterns."""
    
    print(f"\n{'='*80}")
    print("PATTERN WEAKNESS ANALYSIS") 
    print(f"{'='*80}")
    
    # Test cases designed to break your patterns
    challenging_cases = [
        # Hyphenation variations
        ("rt-PA wurde gegeben", "medication", "rtpa", "Hyphenated form not caught"),
        ("Allgemein-Anästhesie durchgeführt", "anesthesia", "allgemeinanästhesie", "Hyphenated anesthesia"),
        
        # Abbreviations
        ("GA eingeleitet", "anesthesia", "ga", "General anesthesia abbreviation"),
        ("ITN durchgeführt", "anesthesia", "itn", "Intubation abbreviation"),
        
        # Negations that should NOT match
        ("Keine rtPA Gabe erfolgt", "medication", None, "Negated medication"),
        ("SOFIA Katheter nicht verwendet", "device", None, "Negated device"),
        
        # Ambiguous cases
        ("Patientin Sofia behandelt", "device", None, "Person name vs device"),
        ("Dr. Trevo operierte", "device", None, "Doctor name vs device"),
        
        # Typos and variations
        ("rtpa verabereicht", "medication", "rtpa", "Typo in context"),
        ("Vollnarkose mit Sevofluran", "anesthesia", "vollnarkose", "Additional context"),
        
        # Complex medical context
        ("TICI 2c nach Stentretriever", "tici_score", "tici 2c", "Complex TICI score"),
        ("prä-interventionell rtPA", "medication", "rtpa", "Complex context"),
    ]
    
    extractor = DebugKeywordExtractor()
    
    print(f"\nTesting {len(challenging_cases)} challenging cases...\n")
    
    issues_found = defaultdict(list)
    
    for i, (text, category, expected, description) in enumerate(challenging_cases):
        print(f"Test {i+1}: {description}")
        print(f"  Text: '{text}'")
        print(f"  Expected: {expected}")
        
        results = extractor.debug_extract_category(text, category)
        found_terms = [r.matched_text for r in results if r.confidence > 0.5]
        
        print(f"  Found: {found_terms}")
        
        # Check if expectation matches reality
        if expected is None:  # Should NOT find anything
            if found_terms:
                issues_found['false_positives'].append({
                    'text': text,
                    'found': found_terms,
                    'description': description
                })
                print(f"  ❌ FALSE POSITIVE: Found {found_terms} but should find nothing")
            else:
                print(f"  ✅ CORRECT: No extraction (as expected)")
        else:  # Should find the expected term
            if expected in [term.lower() for term in found_terms]:
                print(f"  ✅ CORRECT: Found expected term")
            else:
                issues_found['false_negatives'].append({
                    'text': text,
                    'expected': expected,
                    'found': found_terms,
                    'description': description
                })
                print(f"  ❌ FALSE NEGATIVE: Expected '{expected}' but found {found_terms}")
        
        print()
    
    # Summary of issues
    print(f"{'='*60}")
    print("ISSUES SUMMARY")
    print(f"{'='*60}")
    
    if issues_found['false_positives']:
        print(f"\n🚨 FALSE POSITIVES ({len(issues_found['false_positives'])}):")
        for fp in issues_found['false_positives']:
            print(f"  • {fp['description']}: Found '{fp['found']}' in '{fp['text']}'")
    
    if issues_found['false_negatives']: 
        print(f"\n🚨 FALSE NEGATIVES ({len(issues_found['false_negatives'])}):")
        for fn in issues_found['false_negatives']:
            print(f"  • {fn['description']}: Expected '{fn['expected']}' in '{fn['text']}'")
    
    return issues_found

def generate_improvement_plan(issues_found):
    """Generate specific improvement recommendations."""
    
    print(f"\n{'='*80}")
    print("IMPROVEMENT RECOMMENDATIONS")
    print(f"{'='*80}")
    
    print(f"\n🎯 PRIORITY FIXES:")
    
    # Analyze false negatives to suggest pattern improvements
    fn_by_type = defaultdict(list)
    for fn in issues_found.get('false_negatives', []):
        if 'hyphen' in fn['description'].lower():
            fn_by_type['hyphenation'].append(fn)
        elif 'abbreviation' in fn['description'].lower():
            fn_by_type['abbreviations'].append(fn)
        elif 'typo' in fn['description'].lower():
            fn_by_type['typos'].append(fn)
    
    # Analyze false positives to suggest filtering
    fp_by_type = defaultdict(list)
    for fp in issues_found.get('false_positives', []):
        if 'negat' in fp['description'].lower():
            fp_by_type['negation'].append(fp)
        elif 'name' in fp['description'].lower():
            fp_by_type['ambiguity'].append(fp)
    
    print(f"\n1. PATTERN IMPROVEMENTS:")
    
    if fn_by_type['hyphenation']:
        print(f"   • Add hyphenated patterns:")
        print(f"     - r'\\brt-?pa\\b' (matches both 'rtpa' and 'rt-PA')")
        print(f"     - r'\\ballgemein-?anästhesie\\b'")
    
    if fn_by_type['abbreviations']:
        print(f"   • Add abbreviation patterns:")
        print(f"     - r'\\bga\\b' for general anesthesia")
        print(f"     - r'\\bitn\\b' for intubation")
    
    print(f"\n2. FILTERING IMPROVEMENTS:")
    
    if fp_by_type['negation']:
        print(f"   • Implement negation detection:")
        print(f"     - Check for 'keine', 'nicht', 'ohne' before medical terms")
        print(f"     - Example: if re.search(r'kein[e]?\\s+.*?' + term, text):")
    
    if fp_by_type['ambiguity']:
        print(f"   • Add context disambiguation:")
        print(f"     - For 'Sofia': check if preceded by 'Patient', 'Frau', etc.")
        print(f"     - For device names: verify medical context")
    
    print(f"\n3. CONFIDENCE SCORING:")
    print(f"   • High confidence (>0.9): Clear medical context")
    print(f"   • Medium confidence (0.5-0.9): Ambiguous context")
    print(f"   • Low confidence (<0.5): Likely false positive")
    
    print(f"\n4. NEXT STEPS:")
    print(f"   1. Implement negation detection first (biggest impact)")
    print(f"   2. Add hyphenated pattern variants")
    print(f"   3. Build context scoring system")
    print(f"   4. Create more test cases")
    print(f"   5. Measure improvement with F1 scores")

def main_debug_session():
    """Run the complete debugging session."""
    
    print("🩺 STROKE REPORT NLP - DEBUG SESSION")
    print("Let's systematically debug your extraction system...")
    
    # Step 1: Test current system on known cases
    print(f"\n{'='*20} STEP 1: BASIC FUNCTIONALITY {'='*20}")
    test_current_vs_debug_extractor()
    
    # Step 2: Find pattern weaknesses
    print(f"\n{'='*20} STEP 2: PATTERN ANALYSIS {'='*20}")
    issues = analyze_pattern_weaknesses()
    
    # Step 3: Generate improvement plan
    print(f"\n{'='*20} STEP 3: IMPROVEMENT PLAN {'='*20}")
    generate_improvement_plan(issues)
    
    print(f"\n🎓 LEARNING OUTCOMES:")
    print(f"✅ You now see exactly where your patterns fail")
    print(f"✅ You understand why they fail (negation, ambiguity, etc.)")
    print(f"✅ You have a concrete plan to improve them")
    print(f"✅ You can measure improvement systematically")
    
    print(f"\n🚀 HOMEWORK:")
    print(f"1. Implement ONE improvement (start with negation detection)")
    print(f"2. Re-run this debug session to see the impact")
    print(f"3. Add 5 new test cases based on real stroke reports")
    print(f"4. Document what you learned in your own words")

if __name__ == "__main__":
    main_debug_session()