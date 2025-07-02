#!/usr/bin/env python3
"""
Unit tests for accuracy metrics and evaluation components.

Tests the evaluation framework including metrics calculation,
model validation, and performance analysis.
"""

import unittest
import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path
import tempfile
import json

# Add the parent directory to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.metrics import AccuracyMetrics
from evaluation.validation import ModelValidator, create_sample_validation_set
from extractor.keyword_rules import KeywordExtractor
from extractor.preprocessing import TextPreprocessor

class TestAccuracyMetrics(unittest.TestCase):
    """Test cases for the AccuracyMetrics class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.metrics_calculator = AccuracyMetrics()
        
        # Create sample prediction and ground truth data
        self.sample_predictions = pd.DataFrame([
            {
                'report_id': 'test_001',
                'anesthesia': 'allgemeinanästhesie',
                'medication': 'rtpa',
                'device': 'trevo',
                'tici_score': 'tici 3',
                'times': '08:30',
                'complications': None
            },
            {
                'report_id': 'test_002',
                'anesthesia': 'sedierung',
                'medication': 'heparin',
                'device': 'sofia',
                'tici_score': 'tici 2b',
                'times': '09:45',
                'complications': 'blutung'
            },
            {
                'report_id': 'test_003',
                'anesthesia': None,  # Missing prediction
                'medication': 'urokinase',
                'device': 'solitaire',
                'tici_score': 'tici 1',
                'times': '14:15',
                'complications': 'perforation'
            }
        ])
        
        self.sample_ground_truth = pd.DataFrame([
            {
                'report_id': 'test_001',
                'anesthesia': 'allgemeinanästhesie',  # Exact match
                'medication': 'rtpa',  # Exact match
                'device': 'trevo',  # Exact match
                'tici_score': 'tici 3',  # Exact match
                'times': '08:30',  # Exact match
                'complications': None  # Exact match (both null)
            },
            {
                'report_id': 'test_002',
                'anesthesia': 'sedierung',  # Exact match
                'medication': 'heparin',  # Exact match
                'device': 'sofia',  # Exact match
                'tici_score': 'tici 2a',  # Mismatch (predicted 2b, actual 2a)
                'times': '09:45',  # Exact match
                'complications': 'blutung'  # Exact match
            },
            {
                'report_id': 'test_003',
                'anesthesia': 'lokalanästhesie',  # False negative (not predicted)
                'medication': 'urokinase',  # Exact match
                'device': 'solitaire',  # Exact match
                'tici_score': 'tici 1',  # Exact match
                'times': '14:15',  # Exact match
                'complications': 'perforation'  # Exact match
            }
        ])
    
    def test_calculate_binary_metrics_perfect(self):
        """Test binary metrics calculation with perfect predictions."""
        y_true = [1, 1, 0, 0]
        y_pred = [1, 1, 0, 0]
        
        metrics = self.metrics_calculator.calculate_binary_metrics(y_true, y_pred)
        
        self.assertEqual(metrics['precision'], 1.0)
        self.assertEqual(metrics['recall'], 1.0)
        self.assertEqual(metrics['f1'], 1.0)
    
    def test_calculate_binary_metrics_with_errors(self):
        """Test binary metrics with some prediction errors."""
        y_true = [1, 1, 0, 0, 1]
        y_pred = [1, 0, 0, 1, 1]  # One false negative, one false positive
        
        metrics = self.metrics_calculator.calculate_binary_metrics(y_true, y_pred)
        
        # Should have precision = 2/3, recall = 2/3, f1 = 2/3
        self.assertAlmostEqual(metrics['precision'], 0.6667, places=3)
        self.assertAlmostEqual(metrics['recall'], 0.6667, places=3)
        self.assertAlmostEqual(metrics['f1'], 0.6667, places=3)
    
    def test_calculate_binary_metrics_empty(self):
        """Test binary metrics with empty input."""
        metrics = self.metrics_calculator.calculate_binary_metrics([], [])
        
        self.assertEqual(metrics['precision'], 0.0)
        self.assertEqual(metrics['recall'], 0.0)
        self.assertEqual(metrics['f1'], 0.0)
    
    def test_evaluate_extraction_accuracy(self):
        """Test extraction accuracy evaluation."""
        metrics = self.metrics_calculator.evaluate_extraction_accuracy(
            self.sample_predictions, self.sample_ground_truth
        )
        
        # Check that all categories are evaluated
        expected_categories = ['anesthesia', 'medication', 'device', 'tici_score', 'times', 'complications']
        for category in expected_categories:
            self.assertIn(category, metrics)
        
        # Check overall metrics
        self.assertIn('overall', metrics)
        self.assertIn('avg_precision', metrics['overall'])
        self.assertIn('avg_recall', metrics['overall'])
        self.assertIn('avg_f1', metrics['overall'])
        
        # Test specific category - anesthesia has one false negative
        anesthesia_metrics = metrics['anesthesia']
        self.assertIn('precision', anesthesia_metrics)
        self.assertIn('recall', anesthesia_metrics)
        self.assertIn('exact_accuracy', anesthesia_metrics)
        
        # Anesthesia: 2 predicted, 3 actual → recall should be 2/3
        self.assertAlmostEqual(anesthesia_metrics['recall'], 0.6667, places=3)
    
    def test_generate_confusion_matrix(self):
        """Test confusion matrix generation."""
        cm = self.metrics_calculator.generate_confusion_matrix(
            self.sample_predictions, self.sample_ground_truth, 'anesthesia'
        )
        
        # Should be a 2x2 matrix
        self.assertEqual(cm.shape, (2, 2))
        
        # Check that it's a valid confusion matrix (non-negative integers)
        self.assertTrue(np.all(cm >= 0))
        self.assertTrue(np.all(cm == cm.astype(int)))
    
    def test_generate_detailed_report(self):
        """Test detailed report generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "test_report.json")
            
            report = self.metrics_calculator.generate_detailed_report(
                self.sample_predictions, self.sample_ground_truth, output_path
            )
            
            # Check report structure
            self.assertIn('evaluation_summary', report)
            self.assertIn('accuracy_metrics', report)
            self.assertIn('error_analysis', report)
            self.assertIn('recommendations', report)
            
            # Check that file was created
            self.assertTrue(os.path.exists(output_path))
            
            # Verify JSON is valid
            with open(output_path, 'r') as f:
                loaded_report = json.load(f)
            self.assertEqual(report['evaluation_summary']['total_reports'], 
                           loaded_report['evaluation_summary']['total_reports'])
    
    def test_analyze_common_errors(self):
        """Test error analysis functionality."""
        error_analysis = self.metrics_calculator.analyze_common_errors(
            self.sample_predictions, self.sample_ground_truth
        )
        
        # Check error structure
        self.assertIn('false_positives', error_analysis)
        self.assertIn('false_negatives', error_analysis)
        self.assertIn('misclassifications', error_analysis)
        
        # Should detect the anesthesia false negative
        false_negatives = error_analysis['false_negatives']
        anesthesia_fn = [err for err in false_negatives if 'anesthesia:test_003' in err]
        self.assertEqual(len(anesthesia_fn), 1)
        
        # Should detect the TICI score misclassification
        misclassifications = error_analysis['misclassifications']
        tici_misc = [err for err in misclassifications if 'tici_score:test_002' in err]
        self.assertEqual(len(tici_misc), 1)
    
    def test_generate_recommendations(self):
        """Test recommendation generation."""
        # Create metrics with low performance to trigger recommendations
        low_performance_metrics = {
            'anesthesia': {'precision': 0.5, 'recall': 0.6},
            'medication': {'precision': 0.9, 'recall': 0.7},
            'overall': {'avg_f1': 0.65}
        }
        
        recommendations = self.metrics_calculator.generate_recommendations(low_performance_metrics)
        
        # Should generate recommendations for low performance
        self.assertIsInstance(recommendations, list)
        self.assertGreater(len(recommendations), 0)
        
        # Should suggest improving anesthesia precision
        anesthesia_precision_rec = [r for r in recommendations if 'anesthesia precision' in r]
        self.assertGreater(len(anesthesia_precision_rec), 0)

class TestModelValidator(unittest.TestCase):
    """Test cases for the ModelValidator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.validator = ModelValidator()
        
        # Create sample validation data
        self.reports_df, self.truth_df = create_sample_validation_set()
    
    def test_k_fold_validation(self):
        """Test K-fold cross-validation."""
        # Run 3-fold validation
        cv_results = self.validator.k_fold_validation(self.reports_df, self.truth_df, k=3)
        
        # Check result structure
        self.assertIn('summary_stats', cv_results)
        self.assertIn('fold_details', cv_results)
        self.assertIn('total_reports', cv_results)
        self.assertIn('folds', cv_results)
        
        # Check summary statistics
        summary = cv_results['summary_stats']
        for metric in ['precision', 'recall', 'f1', 'exact_accuracy']:
            self.assertIn(metric, summary)
            self.assertIn('mean', summary[metric])
            self.assertIn('std', summary[metric])
            self.assertIn('values', summary[metric])
        
        # Check that we have results for all folds
        self.assertEqual(len(cv_results['fold_details']), 3)
        self.assertEqual(cv_results['folds'], 3)
        
        # Validate metric ranges
        for metric in ['precision', 'recall', 'f1']:
            mean_value = summary[metric]['mean']
            self.assertGreaterEqual(mean_value, 0.0)
            self.assertLessEqual(mean_value, 1.0)
    
    def test_validate_on_test_set(self):
        """Test validation on a test set."""
        validation_results = self.validator.validate_on_test_set(self.reports_df, self.truth_df)
        
        # Check result structure
        self.assertIn('metrics', validation_results)
        self.assertIn('detailed_report', validation_results)
        self.assertIn('predictions', validation_results)
        self.assertIn('test_size', validation_results)
        
        # Check that predictions DataFrame has expected structure
        predictions = validation_results['predictions']
        self.assertIsInstance(predictions, pd.DataFrame)
        self.assertGreater(len(predictions), 0)
        self.assertIn('report_id', predictions.columns)
        
        # Check test size matches
        self.assertEqual(validation_results['test_size'], len(self.reports_df))
    
    def test_generate_validation_report(self):
        """Test validation report generation."""
        # First run validation to get results
        cv_results = self.validator.k_fold_validation(self.reports_df, self.truth_df, k=3)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "validation_report.json")
            
            report = self.validator.generate_validation_report(cv_results, output_path)
            
            # Check report structure
            self.assertIn('validation_summary', report)
            self.assertIn('performance_summary', report)
            self.assertIn('fold_details', report)
            self.assertIn('recommendations', report)
            
            # Check that file was created
            self.assertTrue(os.path.exists(output_path))
    
    def test_validation_recommendations(self):
        """Test validation recommendation generation."""
        # Create sample results with specific performance characteristics
        sample_results = {
            'summary_stats': {
                'f1': {'mean': 0.65, 'std': 0.15},  # Low F1 with high variance
                'precision': {'mean': 0.9, 'std': 0.05},  # High precision
                'recall': {'mean': 0.5, 'std': 0.08},  # Low recall
                'exact_accuracy': {'mean': 0.6, 'std': 0.1}  # Low exact accuracy
            }
        }
        
        recommendations = self.validator._generate_validation_recommendations(sample_results)
        
        # Should generate recommendations for low F1
        f1_recommendations = [r for r in recommendations if 'F1 score is low' in r]
        self.assertGreater(len(f1_recommendations), 0)
        
        # Should suggest improving patterns due to precision-recall imbalance
        balance_recommendations = [r for r in recommendations if 'pattern variations' in r]
        self.assertGreater(len(balance_recommendations), 0)

class TestIntegrationAccuracyMetrics(unittest.TestCase):
    """Integration tests for the complete accuracy evaluation pipeline."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.preprocessor = TextPreprocessor()
        self.extractor = KeywordExtractor()
        self.metrics_calculator = AccuracyMetrics()
        
        # Create realistic test data
        self.test_reports = pd.DataFrame([
            {
                'report_id': 'integration_001',
                'text': '''Patient mit akutem Schlaganfall. Allgemeinanästhesie verwendet.
                          Beginn der Intervention: 08:30 Uhr. rtPA bereits verabreicht.
                          Mechanische Thrombektomie mit Trevo Stentretriever.
                          TICI 3 Rekanalisierung erreicht. Keine Komplikationen.'''
            },
            {
                'report_id': 'integration_002',
                'text': '''67-jähriger Patient. Sedierung für die Prozedur.
                          Start: 09:45 Uhr. SOFIA Aspiration System verwendet.
                          Heparin zur Antikoagulation. Leichte Blutung beobachtet.
                          TICI 2b Ergebnis erreicht.'''
            }
        ])
        
        self.test_ground_truth = pd.DataFrame([
            {
                'report_id': 'integration_001',
                'anesthesia': 'allgemeinanästhesie',
                'medication': 'rtpa',
                'device': 'trevo',
                'treatment_method': 'mechanische thrombektomie',
                'tici_score': 'tici 3',
                'times': '08:30',
                'complications': None
            },
            {
                'report_id': 'integration_002',
                'anesthesia': 'sedierung',
                'medication': 'heparin',
                'device': 'sofia',
                'treatment_method': 'aspiration',
                'tici_score': 'tici 2b',
                'times': '09:45',
                'complications': 'blutung'
            }
        ])
    
    def test_full_evaluation_pipeline(self):
        """Test the complete evaluation pipeline from text to metrics."""
        
        # Step 1: Extract information using rule-based model
        predictions = []
        for _, row in self.test_reports.iterrows():
            text = row['text']
            report_id = row['report_id']
            
            # Preprocess and extract
            cleaned_text = self.preprocessor.clean_text(text)
            extracted = self.extractor.extract_all(cleaned_text, report_id)
            predictions.append(extracted)
        
        predictions_df = pd.DataFrame(predictions)
        
        # Step 2: Calculate accuracy metrics
        metrics = self.metrics_calculator.evaluate_extraction_accuracy(
            predictions_df, self.test_ground_truth
        )
        
        # Step 3: Validate results
        # Should have high accuracy on this controlled test
        overall_f1 = metrics['overall']['avg_f1']
        self.assertGreater(overall_f1, 0.7, "Overall F1 should be reasonably high on controlled test")
        
        # Should extract most categories correctly
        categories_with_data = [cat for cat in metrics.keys() 
                              if cat != 'overall' and metrics[cat].get('total_samples', 0) > 0]
        self.assertGreater(len(categories_with_data), 3, "Should extract multiple categories")
        
        # Test specific category performance
        if 'anesthesia' in metrics:
            anesthesia_accuracy = metrics['anesthesia'].get('exact_accuracy', 0)
            self.assertGreater(anesthesia_accuracy, 0.8, "Anesthesia extraction should be accurate")
    
    def test_error_analysis_integration(self):
        """Test error analysis on realistic data."""
        
        # Create predictions with known errors
        predictions_with_errors = pd.DataFrame([
            {
                'report_id': 'integration_001',
                'anesthesia': 'sedierung',  # Wrong prediction
                'medication': 'rtpa',  # Correct
                'device': None,  # Missing prediction
                'tici_score': 'tici 3',  # Correct
                'times': '08:30',  # Correct
                'complications': 'blutung'  # False positive
            }
        ])
        
        ground_truth_subset = self.test_ground_truth.iloc[:1].copy()
        
        # Analyze errors
        error_analysis = self.metrics_calculator.analyze_common_errors(
            predictions_with_errors, ground_truth_subset
        )
        
        # Should detect misclassification (anesthesia)
        misclassifications = error_analysis['misclassifications']
        self.assertGreater(len(misclassifications), 0)
        
        # Should detect false negative (device)
        false_negatives = error_analysis['false_negatives']
        device_fn = [err for err in false_negatives if 'device:' in err]
        self.assertGreater(len(device_fn), 0)
        
        # Should detect false positive (complications)
        false_positives = error_analysis['false_positives']
        complications_fp = [err for err in false_positives if 'complications:' in err]
        self.assertGreater(len(complications_fp), 0)

# Test runner functions
def run_accuracy_tests():
    """Run all accuracy metric tests."""
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestAccuracyMetrics,
        TestModelValidator,
        TestIntegrationAccuracyMetrics
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"ACCURACY METRICS TESTS SUMMARY")
    print(f"{'='*60}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print(f"\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}")
    
    if result.errors:
        print(f"\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}")
    
    success_rate = ((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun) * 100
    print(f"\nSuccess rate: {success_rate:.1f}%")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    # Run tests when script is executed directly
    print("🧪 Running Accuracy Metrics Tests")
    print("=" * 60)
    
    success = run_accuracy_tests()
    
    if success:
        print("\n🎉 All accuracy metric tests passed!")
    else:
        print("\n❌ Some accuracy metric tests failed!")
        sys.exit(1)