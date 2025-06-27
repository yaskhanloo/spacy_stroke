# ===== tests/test_keyword_extractor.py =====
import unittest
import sys
import os

# Add the parent directory to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from extractor.keyword_rules import KeywordExtractor
from extractor.preprocessing import TextPreprocessor

class TestKeywordExtractor(unittest.TestCase):
    """Test cases for the KeywordExtractor class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.extractor = KeywordExtractor()
        self.preprocessor = TextPreprocessor()
    
    def test_extract_anesthesia_simple(self):
        """Test basic anesthesia extraction."""
        text = "Patient wurde in Allgemeinanästhesie behandelt"
        results = self.extractor.extract_category(text, 'anesthesia')
        
        # Should find exactly one result
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].value, 'allgemeinanästhesie')
    
    def test_extract_anesthesia_multiple(self):
        """Test multiple anesthesia types in one text."""
        text = "Erst Sedierung, dann Allgemeinanästhesie"
        results = self.extractor.extract_category(text, 'anesthesia')
        
        # Should find two results
        self.assertEqual(len(results), 2)
        values = [r.value for r in results]
        self.assertIn('sedierung', values)
        self.assertIn('allgemeinanästhesie', values)
    
    def test_extract_anesthesia_case_insensitive(self):
        """Test that extraction works regardless of case."""
        text = "ALLGEMEINANÄSTHESIE verwendet"
        results = self.extractor.extract_category(text, 'anesthesia')
        
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].value.lower(), 'allgemeinanästhesie')
    
    def test_extract_medication(self):
        """Test medication extraction."""
        text = "rtPA wurde um 07:45 verabreicht, zusätzlich Heparin"
        results = self.extractor.extract_category(text, 'medication')
        
        self.assertEqual(len(results), 2)
        values = [r.value for r in results]
        self.assertIn('rtpa', values)
        self.assertIn('heparin', values)
    
    def test_extract_devices(self):
        """Test medical device extraction."""
        text = "Verwendung des Trevo Systems mit SOFIA Katheter"
        results = self.extractor.extract_category(text, 'device')
        
        self.assertEqual(len(results), 2)
        values = [r.value for r in results]
        self.assertIn('trevo', values)
        self.assertIn('sofia', values)
    
    def test_extract_tici_scores(self):
        """Test TICI score extraction."""
        test_cases = [
            ("TICI 3 erreicht", "tici 3"),
            ("TICI 2b Ergebnis", "tici 2b"),
            ("TICI Score 1", "tici score 1"),
            ("tici3", "tici3")  # Without space
        ]
        
        for text, expected in test_cases:
            results = self.extractor.extract_category(text, 'tici_score')
            self.assertEqual(len(results), 1, f"Failed for text: {text}")
            self.assertEqual(results[0].value.lower(), expected)
    
    def test_extract_times(self):
        """Test time extraction."""
        test_cases = [
            ("Beginn um 08:32 Uhr", "08:32"),
            ("Start: 09:15", "09:15"),
            ("14:30 Intervention", "14:30"),
            ("Beginn: 10:20", "10:20")
        ]
        
        for text, expected in test_cases:
            results = self.extractor.extract_times(text)
            self.assertEqual(len(results), 1, f"Failed for text: {text}")
            self.assertEqual(results[0].value, expected)
    
    def test_extract_complications(self):
        """Test complication extraction."""
        text = "Perforation aufgetreten, leichte Blutung beobachtet"
        results = self.extractor.extract_category(text, 'complications')
        
        self.assertEqual(len(results), 2)
        values = [r.value for r in results]
        self.assertIn('perforation', values)
        self.assertIn('blutung', values)
    
    def test_extract_all_comprehensive(self):
        """Test complete extraction on a full report."""
        text = """
        Patient wurde in Allgemeinanästhesie behandelt. Beginn um 08:32 Uhr.
        rtPA verabreicht. Trevo Stentretriever verwendet.
        TICI 3 erreicht. Keine Komplikationen.
        """
        
        results = self.extractor.extract_all(text, "test_report")
        
        # Check that we extracted something from each category
        self.assertEqual(results['report_id'], "test_report")
        self.assertIsNotNone(results['anesthesia'])
        self.assertIsNotNone(results['medication'])
        self.assertIsNotNone(results['device'])
        self.assertIsNotNone(results['tici_score'])
        self.assertIsNotNone(results['times'])
        
        # Check specific values
        self.assertEqual(results['anesthesia'], 'allgemeinanästhesie')
        self.assertEqual(results['medication'], 'rtpa')
        self.assertEqual(results['device'], 'trevo')
        self.assertEqual(results['tici_score'], 'tici 3')
        self.assertEqual(results['times'], '08:32')
    
    def test_extract_empty_text(self):
        """Test extraction on empty text."""
        results = self.extractor.extract_all("", "empty_report")
        
        # All categories should be None for empty text
        categories = ['anesthesia', 'medication', 'device', 'tici_score', 'complications']
        for category in categories:
            self.assertIsNone(results[category])
    
    def test_extract_no_matches(self):
        """Test extraction when no patterns match."""
        text = "Dies ist ein normaler deutscher Text ohne medizinische Begriffe."
        results = self.extractor.extract_all(text, "no_match_report")
        
        # Should find no medical terms
        categories = ['anesthesia', 'medication', 'device', 'tici_score', 'complications']
        for category in categories:
            self.assertIsNone(results[category])


class TestTextPreprocessor(unittest.TestCase):
    """Test cases for the TextPreprocessor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.preprocessor = TextPreprocessor()
    
    def test_clean_text_basic(self):
        """Test basic text cleaning."""
        text = "  PATIENT wurde behandelt.  "
        cleaned = self.preprocessor.clean_text(text)
        
        # Should be lowercase and trimmed
        self.assertEqual(cleaned, "patient wurde behandelt.")
    
    def test_clean_text_whitespace(self):
        """Test excessive whitespace removal."""
        text = "Patient    wurde     behandelt"
        cleaned = self.preprocessor.clean_text(text)
        
        # Multiple spaces should become single spaces
        self.assertEqual(cleaned, "patient wurde behandelt")
    
    def test_clean_text_headers(self):
        """Test header/footer removal."""
        text = "Befund: Patient wurde behandelt. Dr. Schmidt"
        cleaned = self.preprocessor.clean_text(text)
        
        # Should remove "Befund:" and doctor names
        self.assertNotIn("befund:", cleaned)
        self.assertNotIn("dr. schmidt", cleaned)
    
    def test_tokenize_and_tag(self):
        """Test spaCy tokenization."""
        text = "Patient wurde behandelt"
        doc = self.preprocessor.tokenize_and_tag(text)
        
        # Should return a spaCy Doc object
        self.assertEqual(str(type(doc)), "<class 'spacy.tokens.doc.Doc'>")
        self.assertEqual(len(doc), 3)  # 3 tokens
    
    def test_extract_sentences(self):
        """Test sentence extraction."""
        text = "Patient wurde behandelt. Keine Komplikationen aufgetreten."
        sentences = self.preprocessor.extract_sentences(text)
        
        # Should extract 2 sentences
        self.assertEqual(len(sentences), 2)
        self.assertTrue(sentences[0].startswith("Patient"))
        self.assertTrue(sentences[1].startswith("Keine"))


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete pipeline."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.preprocessor = TextPreprocessor()
        self.extractor = KeywordExtractor()
    
    def test_full_pipeline(self):
        """Test the complete preprocessing + extraction pipeline."""
        raw_text = """
        BEFUND: Patient wurde in ALLGEMEINANÄSTHESIE behandelt.
        Beginn: 08:32 Uhr. rtPA verabreicht.
        Trevo System verwendet. TICI 3 erreicht.
        Dr. Schmidt, Oberarzt
        """
        
        # Process through full pipeline
        cleaned = self.preprocessor.clean_text(raw_text)
        results = self.extractor.extract_all(cleaned, "integration_test")
        
        # Should extract key information despite headers/formatting
        self.assertIsNotNone(results['anesthesia'])
        self.assertIsNotNone(results['medication'])
        self.assertIsNotNone(results['device'])
        self.assertIsNotNone(results['tici_score'])
        self.assertIsNotNone(results['times'])
    
    def test_real_report_sample(self):
        """Test on a realistic German stroke report."""
        report = """
        64-jähriger Patient mit akutem Schlaganfall. Interventionsbeginn: 09:15 Uhr.
        Sedierung für die Prozedur. Mechanische Thrombektomie mit Solitaire Stentretriever.
        Zusätzlich SOFIA Aspiration. rtPA bereits präklinisch verabreicht.
        Komplikationsloser Verlauf. Finales TICI 2b Ergebnis.
        Kein postinterventionelles Hämatom.
        """
        
        cleaned = self.preprocessor.clean_text(report)
        results = self.extractor.extract_all(cleaned, "real_sample")
        
        # Verify realistic extractions
        self.assertEqual(results['anesthesia'], 'sedierung')
        self.assertEqual(results['medication'], 'rtpa')
        self.assertIn('solitaire', results['device'])
        self.assertEqual(results['tici_score'], 'tici 2b')
        self.assertEqual(results['times'], '09:15')


# ===== Test Runner =====
def run_tests():
    """Run all unit tests."""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [TestKeywordExtractor, TestTextPreprocessor, TestIntegration]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"TESTS SUMMARY")
    print(f"{'='*50}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print(f"\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print(f"\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    success_rate = ((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun) * 100
    print(f"\nSuccess rate: {success_rate:.1f}%")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    # Run tests when script is executed directly
    success = run_tests()
    
    if success:
        print("\n🎉 All tests passed!")
    else:
        print("\n❌ Some tests failed!")
        exit(1)


# ===== How to run these tests =====
"""
🚀 RUNNING UNIT TESTS:

1. Save this as 'tests/test_keyword_extractor.py'

2. Run from command line:
   python tests/test_keyword_extractor.py

3. Or run with pytest (install with: pip install pytest):
   pytest tests/

4. For coverage analysis (install with: pip install coverage):
   coverage run tests/test_keyword_extractor.py
   coverage report

WHAT EACH TEST DOES:

✅ test_extract_anesthesia_simple() 
   → Tests if "Allgemeinanästhesie" is correctly found

✅ test_extract_anesthesia_multiple()
   → Tests finding multiple anesthesia types in one text

✅ test_extract_medication()
   → Tests medication extraction like "rtPA", "Heparin"

✅ test_extract_devices()
   → Tests device extraction like "Trevo", "SOFIA"

✅ test_extract_tici_scores()
   → Tests TICI score patterns like "TICI 3", "TICI 2b"

✅ test_extract_times()
   → Tests time extraction like "08:32", "Beginn: 09:15"

✅ test_full_pipeline()
   → Tests the complete process from raw text to results

✅ test_real_report_sample()
   → Tests on realistic German medical report

WHY UNIT TESTS ARE IMPORTANT:

1. 🐛 CATCH BUGS: Find problems before your users do
2. 🔒 PREVENT REGRESSIONS: Make sure fixes don't break other things  
3. 📝 DOCUMENT BEHAVIOR: Tests show how your code should work
4. 🚀 ENABLE REFACTORING: Change code confidently knowing tests will catch issues
5. 🤝 TEAM COLLABORATION: Other developers understand your code through tests

BEST PRACTICES:

• Test edge cases (empty text, no matches, multiple matches)
• Test both positive cases (should find) and negative cases (should not find)
• Keep tests simple and focused on one thing
• Use descriptive test names that explain what you're testing
• Test the behavior, not the implementation
"""