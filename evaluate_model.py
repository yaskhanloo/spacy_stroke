#!/usr/bin/env python3
"""
Model evaluation script for stroke radiology report extraction.

This script provides comprehensive evaluation capabilities including:
- Accuracy metrics calculation
- Model comparison
- Performance visualization
- Detailed error analysis
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import json
import sys
import os
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from extractor.keyword_rules import KeywordExtractor
from extractor.preprocessing import TextPreprocessor
from extractor.ml_model import StrokeMLExtractor
from evaluation.metrics import AccuracyMetrics
from evaluation.validation import ModelValidator

class ModelEvaluator:
    """Comprehensive model evaluation for stroke extraction."""
    
    def __init__(self, models_dir: str = "models/", output_dir: str = "output/"):
        self.models_dir = Path(models_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.preprocessor = TextPreprocessor()
        self.rule_extractor = KeywordExtractor()
        self.ml_extractor = StrokeMLExtractor()
        self.metrics_calculator = AccuracyMetrics()
        self.validator = ModelValidator()
        
        # Load ML models if available
        self._load_models()
    
    def _load_models(self):
        """Load trained ML models if available."""
        if self.models_dir.exists():
            try:
                loaded_models = self.ml_extractor.load_models(str(self.models_dir))
                if loaded_models:
                    print(f"✅ Loaded {len(loaded_models)} ML models")
                else:
                    print("⚠️ No ML models found, using rule-based only")
            except Exception as e:
                print(f"⚠️ Could not load ML models: {e}")
    
    def load_test_data(self, test_reports_path: str, test_ground_truth_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load test dataset."""
        
        test_reports = pd.read_csv(test_reports_path)
        test_ground_truth = pd.read_csv(test_ground_truth_path)
        
        print(f"📊 Loaded test data: {len(test_reports)} reports")
        
        return test_reports, test_ground_truth
    
    def evaluate_rule_based_model(self, test_reports: pd.DataFrame, 
                                 test_ground_truth: pd.DataFrame) -> Dict:
        """Evaluate rule-based extraction model."""
        
        print("📊 Evaluating rule-based model...")
        
        predictions = []
        processing_times = []
        
        for _, row in test_reports.iterrows():
            text = row['text']
            report_id = row['report_id']
            
            # Time the extraction
            import time
            start_time = time.time()
            
            # Preprocess and extract
            cleaned_text = self.preprocessor.clean_text(text)
            extracted = self.rule_extractor.extract_all(cleaned_text, report_id)
            
            processing_time = time.time() - start_time
            processing_times.append(processing_time)
            
            predictions.append(extracted)
        
        predictions_df = pd.DataFrame(predictions)
        
        # Calculate metrics
        metrics = self.metrics_calculator.evaluate_extraction_accuracy(
            predictions_df, test_ground_truth
        )
        
        # Generate detailed report
        detailed_report = self.metrics_calculator.generate_detailed_report(
            predictions_df, test_ground_truth, 
            str(self.output_dir / "rule_based_evaluation_report.json")
        )
        
        return {
            'model_type': 'rule_based',
            'metrics': metrics,
            'detailed_report': detailed_report,
            'predictions': predictions_df,
            'processing_times': processing_times,
            'avg_processing_time': np.mean(processing_times),
            'total_reports': len(predictions)
        }
    
    def evaluate_ml_model(self, test_reports: pd.DataFrame, 
                         test_ground_truth: pd.DataFrame) -> Dict:
        """Evaluate ML-based extraction model."""
        
        if not self.ml_extractor.models:
            return {'error': 'No ML models loaded'}
        
        print("🤖 Evaluating ML model...")
        
        predictions = []
        processing_times = []
        
        for _, row in test_reports.iterrows():
            text = row['text']
            report_id = row['report_id']
            
            # Time the extraction
            import time
            start_time = time.time()
            
            extracted = self.ml_extractor.extract_with_ml(text, report_id)
            
            processing_time = time.time() - start_time
            processing_times.append(processing_time)
            
            predictions.append(extracted)
        
        predictions_df = pd.DataFrame(predictions)
        
        # Calculate metrics
        try:
            metrics = self.metrics_calculator.evaluate_extraction_accuracy(
                predictions_df, test_ground_truth
            )
            
            # Generate detailed report
            detailed_report = self.metrics_calculator.generate_detailed_report(
                predictions_df, test_ground_truth,
                str(self.output_dir / "ml_evaluation_report.json")
            )
            
            return {
                'model_type': 'ml_based',
                'metrics': metrics,
                'detailed_report': detailed_report,
                'predictions': predictions_df,
                'processing_times': processing_times,
                'avg_processing_time': np.mean(processing_times),
                'total_reports': len(predictions)
            }
            
        except Exception as e:
            return {
                'model_type': 'ml_based',
                'error': f'ML evaluation failed: {str(e)}',
                'predictions': predictions_df
            }
    
    def compare_models(self, rule_results: Dict, ml_results: Dict) -> Dict:
        """Compare rule-based and ML model performances."""
        
        print("⚖️ Comparing model performances...")
        
        comparison = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'models_compared': []
        }
        
        # Rule-based metrics
        if 'error' not in rule_results:
            rule_metrics = rule_results['metrics']['overall']
            comparison['rule_based'] = {
                'f1_score': rule_metrics['avg_f1'],
                'precision': rule_metrics['avg_precision'],
                'recall': rule_metrics['avg_recall'],
                'avg_processing_time': rule_results['avg_processing_time'],
                'total_reports': rule_results['total_reports']
            }
            comparison['models_compared'].append('rule_based')
        
        # ML metrics
        if 'error' not in ml_results:
            ml_metrics = ml_results['metrics']['overall']
            comparison['ml_based'] = {
                'f1_score': ml_metrics['avg_f1'],
                'precision': ml_metrics['avg_precision'],
                'recall': ml_metrics['avg_recall'],
                'avg_processing_time': ml_results['avg_processing_time'],
                'total_reports': ml_results['total_reports']
            }
            comparison['models_compared'].append('ml_based')
        
        # Determine best model
        if len(comparison['models_compared']) == 2:
            rule_f1 = comparison['rule_based']['f1_score']
            ml_f1 = comparison['ml_based']['f1_score']
            
            comparison['best_model'] = {
                'name': 'rule_based' if rule_f1 > ml_f1 else 'ml_based',
                'f1_score': max(rule_f1, ml_f1),
                'f1_difference': abs(rule_f1 - ml_f1)
            }
            
            # Performance analysis
            comparison['analysis'] = self._analyze_model_performance(comparison)
        
        elif 'rule_based' in comparison:
            comparison['best_model'] = {
                'name': 'rule_based',
                'f1_score': comparison['rule_based']['f1_score'],
                'note': 'Only rule-based model available'
            }
        
        return comparison
    
    def _analyze_model_performance(self, comparison: Dict) -> Dict:
        """Analyze model performance differences."""
        
        rule_metrics = comparison['rule_based']
        ml_metrics = comparison['ml_based']
        
        analysis = {
            'performance_summary': {},
            'recommendations': []
        }
        
        # F1 Score Analysis
        f1_diff = abs(rule_metrics['f1_score'] - ml_metrics['f1_score'])
        if f1_diff < 0.05:
            analysis['performance_summary']['f1'] = 'Similar performance'
            analysis['recommendations'].append('Consider ensemble approach')
        elif rule_metrics['f1_score'] > ml_metrics['f1_score']:
            analysis['performance_summary']['f1'] = 'Rule-based superior'
            analysis['recommendations'].append('Focus on improving rule patterns')
        else:
            analysis['performance_summary']['f1'] = 'ML model superior'
            analysis['recommendations'].append('Expand ML training data')
        
        # Precision vs Recall Analysis
        rule_precision = rule_metrics['precision']
        rule_recall = rule_metrics['recall']
        ml_precision = ml_metrics['precision']
        ml_recall = ml_metrics['recall']
        
        if rule_precision > ml_precision:
            analysis['performance_summary']['precision'] = 'Rule-based more precise'
        else:
            analysis['performance_summary']['precision'] = 'ML model more precise'
        
        if rule_recall > ml_recall:
            analysis['performance_summary']['recall'] = 'Rule-based better recall'
        else:
            analysis['performance_summary']['recall'] = 'ML model better recall'
        
        # Speed Analysis
        rule_speed = rule_metrics['avg_processing_time']
        ml_speed = ml_metrics['avg_processing_time']
        
        if rule_speed < ml_speed:
            analysis['performance_summary']['speed'] = 'Rule-based faster'
        else:
            analysis['performance_summary']['speed'] = 'ML model faster'
        
        analysis['recommendations'].append(f"Speed difference: {abs(rule_speed - ml_speed):.4f}s per report")
        
        return analysis
    
    def generate_performance_plots(self, rule_results: Dict, ml_results: Dict):
        """Generate performance visualization plots."""
        
        print("📈 Generating performance plots...")
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Stroke Extraction Model Performance Comparison', fontsize=16)
        
        # Plot 1: Overall Metrics Comparison
        if 'error' not in rule_results and 'error' not in ml_results:
            ax1 = axes[0, 0]
            
            metrics = ['F1 Score', 'Precision', 'Recall']
            rule_values = [
                rule_results['metrics']['overall']['avg_f1'],
                rule_results['metrics']['overall']['avg_precision'],
                rule_results['metrics']['overall']['avg_recall']
            ]
            ml_values = [
                ml_results['metrics']['overall']['avg_f1'],
                ml_results['metrics']['overall']['avg_precision'],
                ml_results['metrics']['overall']['avg_recall']
            ]
            
            x = np.arange(len(metrics))
            width = 0.35
            
            ax1.bar(x - width/2, rule_values, width, label='Rule-based', alpha=0.8)
            ax1.bar(x + width/2, ml_values, width, label='ML-based', alpha=0.8)
            
            ax1.set_xlabel('Metrics')
            ax1.set_ylabel('Score')
            ax1.set_title('Overall Performance Metrics')
            ax1.set_xticks(x)
            ax1.set_xticklabels(metrics)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # Plot 2: Category-wise F1 Scores
        ax2 = axes[0, 1]
        
        if 'error' not in rule_results:
            categories = []
            rule_f1_scores = []
            
            for category, metrics in rule_results['metrics'].items():
                if category != 'overall' and 'f1' in metrics:
                    categories.append(category)
                    rule_f1_scores.append(metrics['f1'])
            
            if categories:
                ax2.barh(categories, rule_f1_scores, alpha=0.8, label='Rule-based')
                ax2.set_xlabel('F1 Score')
                ax2.set_title('Category-wise F1 Scores')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
        
        # Plot 3: Processing Time Comparison
        ax3 = axes[1, 0]
        
        models = []
        times = []
        
        if 'error' not in rule_results:
            models.append('Rule-based')
            times.append(rule_results['avg_processing_time'])
        
        if 'error' not in ml_results:
            models.append('ML-based')
            times.append(ml_results['avg_processing_time'])
        
        if models:
            ax3.bar(models, times, alpha=0.8, color=['skyblue', 'lightcoral'])
            ax3.set_ylabel('Time (seconds)')
            ax3.set_title('Average Processing Time per Report')
            ax3.grid(True, alpha=0.3)
        
        # Plot 4: Error Analysis
        ax4 = axes[1, 1]
        
        if 'error' not in rule_results:
            # Get error counts from detailed report
            error_analysis = rule_results['detailed_report']['error_analysis']
            error_types = ['False Positives', 'False Negatives', 'Misclassifications']
            error_counts = [
                len(error_analysis['false_positives']),
                len(error_analysis['false_negatives']),
                len(error_analysis['misclassifications'])
            ]
            
            ax4.pie(error_counts, labels=error_types, autopct='%1.1f%%', startangle=90)
            ax4.set_title('Error Distribution (Rule-based)')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / 'model_performance_comparison.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📊 Performance plots saved: {plot_path}")
    
    def generate_category_analysis(self, rule_results: Dict, ml_results: Dict) -> Dict:
        """Generate detailed category-wise analysis."""
        
        analysis = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'category_performance': {},
            'improvement_suggestions': {}
        }
        
        # Analyze rule-based performance by category
        if 'error' not in rule_results:
            for category, metrics in rule_results['metrics'].items():
                if category != 'overall' and isinstance(metrics, dict):
                    analysis['category_performance'][category] = {
                        'rule_based': {
                            'f1': metrics.get('f1', 0),
                            'precision': metrics.get('precision', 0),
                            'recall': metrics.get('recall', 0),
                            'exact_accuracy': metrics.get('exact_accuracy', 0)
                        }
                    }
                    
                    # Generate improvement suggestions
                    suggestions = []
                    if metrics.get('precision', 0) < 0.8:
                        suggestions.append("Refine extraction patterns to reduce false positives")
                    if metrics.get('recall', 0) < 0.8:
                        suggestions.append("Add more pattern variations to improve recall")
                    if metrics.get('exact_accuracy', 0) < 0.7:
                        suggestions.append("Review pattern accuracy for exact matches")
                    
                    analysis['improvement_suggestions'][category] = suggestions
        
        return analysis
    
    def run_comprehensive_evaluation(self, test_reports_path: str, 
                                   test_ground_truth_path: str,
                                   detailed_report: bool = True) -> Dict:
        """Run comprehensive model evaluation."""
        
        print("🧪 Starting Comprehensive Model Evaluation")
        print("=" * 60)
        
        try:
            # Load test data
            test_reports, test_ground_truth = self.load_test_data(
                test_reports_path, test_ground_truth_path
            )
            
            results = {}
            
            # Evaluate rule-based model
            rule_results = self.evaluate_rule_based_model(test_reports, test_ground_truth)
            results['rule_based'] = rule_results
            
            if 'error' not in rule_results:
                print(f"📊 Rule-based F1: {rule_results['metrics']['overall']['avg_f1']:.3f}")
            
            # Evaluate ML model
            ml_results = self.evaluate_ml_model(test_reports, test_ground_truth)
            results['ml_based'] = ml_results
            
            if 'error' not in ml_results:
                print(f"🤖 ML-based F1: {ml_results['metrics']['overall']['avg_f1']:.3f}")
            
            # Compare models
            comparison = self.compare_models(rule_results, ml_results)
            results['comparison'] = comparison
            
            if 'best_model' in comparison:
                print(f"🏆 Best model: {comparison['best_model']['name']}")
            
            # Generate visualizations
            if detailed_report:
                self.generate_performance_plots(rule_results, ml_results)
                
                # Category analysis
                category_analysis = self.generate_category_analysis(rule_results, ml_results)
                results['category_analysis'] = category_analysis
            
            # Save comprehensive report
            report_path = self.output_dir / 'comprehensive_evaluation_report.json'
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False, default=str)
            
            print(f"\n✅ Evaluation completed successfully!")
            print(f"📄 Comprehensive report: {report_path}")
            print(f"📊 Check {self.output_dir} for detailed results and plots")
            
            return results
            
        except Exception as e:
            print(f"❌ Evaluation failed: {e}")
            raise

def main():
    """Main evaluation script entry point."""
    
    parser = argparse.ArgumentParser(description="Evaluate stroke extraction models")
    parser.add_argument('--test-reports', type=str, required=True, 
                       help='Path to test reports CSV file')
    parser.add_argument('--test-ground-truth', type=str, required=True,
                       help='Path to test ground truth CSV file')
    parser.add_argument('--models-dir', type=str, default='models/',
                       help='Directory containing trained models')
    parser.add_argument('--output-dir', type=str, default='output/',
                       help='Output directory for evaluation results')
    parser.add_argument('--detailed-report', action='store_true',
                       help='Generate detailed analysis and plots')
    parser.add_argument('--rule-only', action='store_true',
                       help='Evaluate only rule-based model')
    parser.add_argument('--ml-only', action='store_true',
                       help='Evaluate only ML model')
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = ModelEvaluator(args.models_dir, args.output_dir)
    
    # Run evaluation
    results = evaluator.run_comprehensive_evaluation(
        args.test_reports,
        args.test_ground_truth,
        args.detailed_report
    )
    
    return results

if __name__ == "__main__":
    # Example usage when run directly
    if len(sys.argv) == 1:
        print("🧪 Running evaluation with sample data...")
        
        # Create sample test data if not exists
        sample_reports = pd.DataFrame([
            {'report_id': 'test_001', 'text': 'Patient mit Allgemeinanästhesie. rtPA verabreicht. Trevo verwendet. TICI 3 erreicht.'},
            {'report_id': 'test_002', 'text': 'Sedierung für Prozedur. SOFIA System. Heparin gegeben. TICI 2b Ergebnis.'}
        ])
        
        sample_ground_truth = pd.DataFrame([
            {'report_id': 'test_001', 'anesthesia': 'allgemeinanästhesie', 'medication': 'rtpa', 'device': 'trevo', 'tici_score': 'tici 3'},
            {'report_id': 'test_002', 'anesthesia': 'sedierung', 'medication': 'heparin', 'device': 'sofia', 'tici_score': 'tici 2b'}
        ])
        
        # Save sample data
        Path("data/test").mkdir(parents=True, exist_ok=True)
        sample_reports.to_csv("data/test/test_reports.csv", index=False)
        sample_ground_truth.to_csv("data/test/test_ground_truth.csv", index=False)
        
        # Run evaluation
        evaluator = ModelEvaluator()
        results = evaluator.run_comprehensive_evaluation(
            "data/test/test_reports.csv",
            "data/test/test_ground_truth.csv",
            detailed_report=True
        )
    else:
        results = main()