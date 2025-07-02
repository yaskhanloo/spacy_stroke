# Hands-On NLP Learning Exercises
# Complete these exercises to understand your stroke report project deeply

import re
import spacy
from typing import List, Dict, Tuple

# ============================================================================
# EXERCISE 1: Understanding Regular Expressions
# ============================================================================

def exercise_1_regex_basics():
    """
    Learn regex by building patterns step by step.
    
    GOAL: Understand how each part of a regex works
    """
    print("=== EXERCISE 1: REGEX BASICS ===\n")
    
    # Sample medical texts to test against
    test_texts = [
        "Patient hatte TICI 3 Ergebnis",
        "TICI 2a erreicht nach Intervention", 
        "Finales tici 1 outcome",
        "TICI score 2b documented",
        "Criticize the TICI approach",  # Should NOT match
        "TICI 4 is not valid",         # Should NOT match
    ]
    
    # TODO: Fill in these patterns and test them
    exercises = [
        {
            'description': 'Match literal "tici" (case insensitive)',
            'pattern': r'\btici\b',
            'should_match': ['TICI', 'tici', 'Tici'],
            'should_not_match': ['criticize', 'ticillin']
        },
        {
            'description': 'Match "tici" followed by a space and a number',
            'pattern': r'\btici\s+\d\b',  # YOUR PATTERN HERE
            'should_match': ['TICI 3', 'tici 2'],
            'should_not_match': ['TICI', 'TICIx']
        },
        {
            'description': 'Match TICI scores: "tici" + optional space + number + optional letter',
            'pattern': r'\btici\s*[0-9][a-z]?\b',  # YOUR PATTERN HERE
            'should_match': ['TICI 3', 'tici2a', 'TICI 1b', 'tici 0'],
            'should_not_match': ['TICI 4', 'criticize']
        }
    ]
    
    # Test your patterns
    for i, exercise in enumerate(exercises, 1):
        print(f"Exercise 1.{i}: {exercise['description']}")
        print(f"Your pattern: {exercise['pattern']}")
        
        if exercise['pattern']:  # Only test if pattern is filled in
            for text in test_texts:
                matches = re.findall(exercise['pattern'], text, re.IGNORECASE)
                print(f"  '{text}' -> {matches}")
        
        print("Expected to match:", exercise['should_match'])
        print("Expected NOT to match:", exercise['should_not_match'])
        print("-" * 50)


def exercise_1_solution():
    """Solution for Exercise 1 - DON'T LOOK UNTIL YOU TRY!"""
    solutions = [
        r'\btici\b',                    # 1.1: Word boundaries prevent partial matches
        r'\btici\s+\d\b',              # 1.2: \s+ = one or more spaces, \d = digit
        r'\btici\s*[0-3][abc]?\b'      # 1.3: \s* = zero or more spaces, [0-3] = 0-3, [abc]? = optional a/b/c
    ]
    return solutions


# ============================================================================
# EXERCISE 2: Text Preprocessing Deep Dive
# ============================================================================

def exercise_2_preprocessing():
    """
    Understand why and how we clean text.
    
    GOAL: See the impact of each preprocessing step
    """
    print("\n=== EXERCISE 2: TEXT PREPROCESSING ===\n")
    
    # Raw medical report with common issues
    raw_text = """
    BEFUND: Patient wurde in    ALLGEMEINANÄSTHESIE behandelt.
    
    Beginn der Intervention um 08:32 UHR. rtPA wurde verabreicht.
    
    Dr. Schmidt, Oberarzt
    Universitätsklinikum Hamburg
    """
    
    print("Original text:")
    print(repr(raw_text))  # repr() shows whitespace characters
    print("\nStep-by-step cleaning:")
    
    # TODO: Implement each cleaning step and observe the effects
    
    # Step 1: Remove excessive whitespace
    step1 = ""  # YOUR CODE HERE: Use re.sub to replace multiple spaces with single space
    print(f"1. After whitespace cleanup: {repr(step1)}")
    
    # Step 2: Remove headers like "BEFUND:"
    step2 = ""  # YOUR CODE HERE: Remove "BEFUND:" pattern
    print(f"2. After header removal: {repr(step2)}")
    
    # Step 3: Remove doctor names
    step3 = ""  # YOUR CODE HERE: Remove "Dr. Schmidt, Oberarzt" pattern
    print(f"3. After doctor name removal: {repr(step3)}")
    
    # Step 4: Convert to lowercase
    step4 = ""  # YOUR CODE HERE: Convert to lowercase
    print(f"4. After lowercase: {repr(step4)}")
    
    # Step 5: Strip extra whitespace
    final = ""  # YOUR CODE HERE: Strip leading/trailing whitespace
    print(f"5. Final result: {repr(final)}")
    
    # Analysis questions:
    print("\n🤔 THINK ABOUT:")
    print("1. What would happen if we skipped the lowercase step?")
    print("2. Why remove doctor names? What problems could they cause?")
    print("3. What other cleaning steps might be useful?")


def exercise_2_solution():
    """Solution for Exercise 2"""
    raw_text = """
    BEFUND: Patient wurde in    ALLGEMEINANÄSTHESIE behandelt.
    
    Beginn der Intervention um 08:32 UHR. rtPA wurde verabreicht.
    
    Dr. Schmidt, Oberarzt
    Universitätsklinikum Hamburg
    """
    
    # Step-by-step solution
    step1 = re.sub(r'\s+', ' ', raw_text)
    step2 = re.sub(r'(Befund|Diagnose|Impression):\s*', '', step1, flags=re.IGNORECASE)
    step3 = re.sub(r'(Dr\.|Prof\.|Oberarzt|Chefarzt)\s+\w+', '', step2)
    step4 = step3.lower()
    final = step4.strip()
    
    return final


# ============================================================================
# EXERCISE 3: Pattern Matching Analysis
# ============================================================================

def exercise_3_pattern_analysis():
    """
    Analyze existing patterns to understand their strengths and weaknesses.
    
    GOAL: Critical thinking about pattern design
    """
    print("\n=== EXERCISE 3: PATTERN ANALYSIS ===\n")
    
    # Current patterns from your project
    patterns = {
        'anesthesia': [r'\ballgemeinanästhesie\b', r'\bsedierung\b', r'\blokalanästhesie\b'],
        'medication': [r'\brtpa\b', r'\burkinas[e]?\b', r'\bheparin\b'],
        'device': [r'\bsofia\b', r'\btrevo\b', r'\bsolitaire\b']
    }
    
    # Test cases - some should match, some shouldn't
    test_cases = [
        "Patient in Allgemeinanästhesie",           # Should match anesthesia
        "Allgemein-Anästhesie verwendet",           # Should this match? Why/why not?
        "Lokalanästhesie am Arm",                   # Should match anesthesia  
        "rtPA verabreicht",                         # Should match medication
        "rt-PA injection",                          # Should this match? Why/why not?
        "Urokinase als Thrombolytikum",            # Should match medication
        "Sofia Katheter eingesetzt",                # Should match device
        "SOFIA aspiration system",                  # Should match device
        "Patient heißt Sofia",                      # Should NOT match device
    ]
    
    # TODO: For each pattern, analyze its performance
    for category, pattern_list in patterns.items():
        print(f"\n--- {category.upper()} PATTERNS ---")
        
        for pattern in pattern_list:
            print(f"\nPattern: {pattern}")
            print("Matches found:")
            
            for test_case in test_cases:
                matches = re.findall(pattern, test_case, re.IGNORECASE)
                if matches:
                    print(f"  ✓ '{test_case}' -> {matches}")
            
            # TODO: Answer these questions for each pattern:
            print("\n🤔 ANALYSIS QUESTIONS:")
            print("1. What are the strengths of this pattern?")
            print("2. What edge cases might it miss?")
            print("3. What false positives might it catch?")
            print("4. How would you improve it?")
            print("-" * 60)


# ============================================================================
# EXERCISE 4: Building Your Own Extractor
# ============================================================================

def exercise_4_build_extractor():
    """
    Build a simple extractor from scratch to understand the full process.
    
    GOAL: Understand how extraction systems work end-to-end
    """
    print("\n=== EXERCISE 4: BUILD YOUR OWN EXTRACTOR ===\n")
    
    # Sample text
    text = """
    64-jähriger Patient mit akutem Schlaganfall. 
    Interventionsbeginn: 09:15 Uhr. 
    Patient ist 78 Jahre alt.
    Behandlung unter Sedierung.
    """
    
    class SimpleExtractor:
        def __init__(self):
            # TODO: Define patterns for age extraction
            self.age_patterns = [
                # Pattern 1: "64-jähriger" format
                r'',  # YOUR PATTERN HERE
                
                # Pattern 2: "78 Jahre alt" format  
                r'',  # YOUR PATTERN HERE
                
                # Pattern 3: Just number + "y" (English abbreviation)
                r'',  # YOUR PATTERN HERE
            ]
        
        def extract_age(self, text: str) -> List[str]:
            """Extract all age mentions from text."""
            ages = []
            
            # TODO: Use your patterns to find ages
            for pattern in self.age_patterns:
                if pattern:  # Only if pattern is defined
                    matches = re.findall(pattern, text, re.IGNORECASE)
                    ages.extend(matches)
            
            return ages
        
        def extract_times(self, text: str) -> List[str]:
            """Extract time mentions (HH:MM format)."""
            # TODO: Write a pattern to match times like "09:15"
            time_pattern = r''  # YOUR PATTERN HERE
            
            if time_pattern:
                return re.findall(time_pattern, text)
            return []
    
    # Test your extractor
    extractor = SimpleExtractor()
    
    print("Testing age extraction:")
    ages = extractor.extract_age(text)
    print(f"Found ages: {ages}")
    
    print("\nTesting time extraction:")
    times = extractor.extract_times(text)
    print(f"Found times: {times}")
    
    # TODO: Evaluate your results
    print("\n🎯 EVALUATION:")
    print("1. Did you find all the ages in the text?")
    print("2. Did you find all the times?") 
    print("3. Any false positives?")
    print("4. How could you improve accuracy?")


def exercise_4_solution():
    """Solution for Exercise 4"""
    age_patterns = [
        r'(\d+)-jährig',                    # Captures "64" from "64-jähriger"
        r'(\d+)\s+Jahre?\s+alt',           # Captures "78" from "78 Jahre alt"
        r'(\d+)y\b'                        # Captures "65" from "65y"
    ]
    
    time_pattern = r'(\d{1,2}:\d{2})'      # Captures HH:MM format
    
    return age_patterns, time_pattern


# ============================================================================
# EXERCISE 5: spaCy Exploration
# ============================================================================

def exercise_5_spacy_exploration():
    """
    Explore spaCy to understand what it can tell us about medical text.
    
    GOAL: Understand linguistic analysis beyond simple pattern matching
    """
    print("\n=== EXERCISE 5: spaCy EXPLORATION ===\n")
    
    # Load German model
    try:
        nlp = spacy.load("de_core_news_sm")
    except OSError:
        print("German spaCy model not found. Please install with:")
        print("python -m spacy download de_core_news_sm")
        return
    
    # Medical text to analyze
    text = "Patient wurde mit rtPA behandelt. Dr. Schmidt führte die Thrombektomie durch."
    
    # Process with spaCy
    doc = nlp(text)
    
    print(f"Original text: {text}\n")
    
    # TODO: Explore different aspects of the linguistic analysis
    
    print("1. TOKENIZATION:")
    # TODO: Print each token
    for token in doc:
        print(f"  '{token.text}'")
    
    print("\n2. PART-OF-SPEECH TAGGING:")
    # TODO: Print each token with its POS tag
    for token in doc:
        print(f"  {token.text} -> {token.pos_} ({spacy.explain(token.pos_)})")
    
    print("\n3. NAMED ENTITY RECOGNITION:")
    # TODO: Print identified entities
    for ent in doc.ents:
        print(f"  {ent.text} -> {ent.label_} ({spacy.explain(ent.label_)})")
    
    print("\n4. NOUN CHUNKS:")
    # TODO: Print noun phrases
    for chunk in doc.noun_chunks:
        print(f"  {chunk.text}")
    
    print("\n5. DEPENDENCY PARSING:")
    # TODO: Print grammatical relationships
    for token in doc:
        print(f"  {token.text} --{token.dep_}--> {token.head.text}")
    
    # Analysis questions
    print("\n🤔 REFLECTION:")
    print("1. What medical terms did spaCy recognize as entities?")
    print("2. What did it miss that your regex patterns would catch?")
    print("3. How could you combine spaCy with regex for better results?")
    print("4. When would you use spaCy vs simple patterns?")


# ============================================================================
# EXERCISE 6: Evaluation and Metrics
# ============================================================================

def exercise_6_evaluation():
    """
    Learn to evaluate your NLP system systematically.
    
    GOAL: Understand precision, recall, and F1-score
    """
    print("\n=== EXERCISE 6: EVALUATION METRICS ===\n")
    
    # Create ground truth data (what SHOULD be extracted)
    test_data = [
        {
            'text': "Patient in Allgemeinanästhesie, rtPA gegeben",
            'true_anesthesia': ['allgemeinanästhesie'],
            'true_medication': ['rtpa']
        },
        {
            'text': "Sedierung verwendet, keine Medikation",
            'true_anesthesia': ['sedierung'],
            'true_medication': []
        },
        {
            'text': "TICI 3 erreicht nach Trevo Intervention",
            'true_anesthesia': [],
            'true_medication': []
        }
    ]
    
    # Simple extractor for testing
    def extract_anesthesia(text):
        pattern = r'\b(allgemeinanästhesie|sedierung|narkose)\b'
        return re.findall(pattern, text.lower())
    
    def extract_medication(text):
        pattern = r'\b(rtpa|heparin|aspirin)\b'
        return re.findall(pattern, text.lower())
    
    # TODO: Calculate metrics
    def calculate_metrics(predicted: List[str], true: List[str]) -> Dict[str, float]:
        """Calculate precision, recall, F1 for one test case."""
        predicted_set = set(predicted)
        true_set = set(true)
        
        # TODO: Calculate these metrics
        true_positives = 0      # Items in both predicted and true
        false_positives = 0     # Items in predicted but not true  
        false_negatives = 0     # Items in true but not predicted
        
        precision = 0.0         # TP / (TP + FP)
        recall = 0.0           # TP / (TP + FN)
        f1 = 0.0               # 2 * (precision * recall) / (precision + recall)
        
        return {
            'precision': precision,
            'recall': recall, 
            'f1': f1,
            'tp': true_positives,
            'fp': false_positives,
            'fn': false_negatives
        }
    
    # Test the system
    print("EVALUATION RESULTS:")
    all_anesthesia_metrics = []
    
    for i, test_case in enumerate(test_data):
        predicted_anesthesia = extract_anesthesia(test_case['text'])
        true_anesthesia = test_case['true_anesthesia']
        
        metrics = calculate_metrics(predicted_anesthesia, true_anesthesia)
        all_anesthesia_metrics.append(metrics)
        
        print(f"\nTest case {i+1}: '{test_case['text']}'")
        print(f"  Predicted: {predicted_anesthesia}")
        print(f"  True: {true_anesthesia}")
        print(f"  Metrics: P={metrics['precision']:.2f}, R={metrics['recall']:.2f}, F1={metrics['f1']:.2f}")
    
    # TODO: Calculate overall metrics (average across test cases)
    print("\n=== OVERALL PERFORMANCE ===")
    print("Calculate and report average precision, recall, F1")


def exercise_6_solution():
    """Solution for metrics calculation"""
    def calculate_metrics(predicted: List[str], true: List[str]) -> Dict[str, float]:
        predicted_set = set(predicted)
        true_set = set(true)
        
        true_positives = len(predicted_set & true_set)
        false_positives = len(predicted_set - true_set)
        false_negatives = len(true_set - predicted_set)
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': true_positives,
            'fp': false_positives,
            'fn': false_negatives
        }
    
    return calculate_metrics


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("🧠 NLP LEARNING EXERCISES")
    print("=" * 50)
    print("Work through these exercises to understand your stroke report project.")
    print("Don't peek at solutions until you've tried each exercise!")
    print()
    
    # Uncomment exercises as you complete them:
    
    exercise_1_regex_basics()
    # exercise_2_preprocessing()  
    # exercise_3_pattern_analysis()
    # exercise_4_build_extractor()
    # exercise_5_spacy_exploration()
    # exercise_6_evaluation()
    
    print("\n🎓 NEXT STEPS:")
    print("1. Complete all exercises")
    print("2. Compare your solutions with the provided solutions")
    print("3. Apply what you learned to improve your stroke report system")
    print("4. Read the learning guide for deeper theoretical understanding")