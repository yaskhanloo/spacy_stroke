import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from extractor.keyword_rules import KeywordExtractor
from extractor.preprocessing import TextPreprocessor
from evaluation.metrics import AccuracyMetrics

class ModelValidator:
    """Validate stroke extraction models using cross-validation and test sets."""
    
    def __init__(self, extractor_class=KeywordExtractor, preprocessor_class=TextPreprocessor):
        self.extractor_class = extractor_class
        self.preprocessor_class = preprocessor_class
        self.metrics_calculator = AccuracyMetrics()
        
    def k_fold_validation(self, reports_df: pd.DataFrame, 
                         ground_truth_df: pd.DataFrame, 
                         k: int = 5) -> Dict[str, List[float]]:
        """
        Perform K-fold cross-validation on the extraction model.
        
        Args:
            reports_df: DataFrame with report texts
            ground_truth_df: DataFrame with ground truth annotations
            k: Number of folds
            
        Returns:
            Dictionary with performance metrics across folds
        """
        
        # Ensure we have matching report IDs
        common_ids = list(set(reports_df['report_id']) & set(ground_truth_df['report_id']))
        if len(common_ids) < k:
            raise ValueError(f"Not enough reports ({len(common_ids)}) for {k}-fold validation")
        
        # Filter to common reports
        reports_subset = reports_df[reports_df['report_id'].isin(common_ids)].copy()
        truth_subset = ground_truth_df[ground_truth_df['report_id'].isin(common_ids)].copy()
        
        # Sort by report_id for consistent ordering
        reports_subset = reports_subset.sort_values('report_id').reset_index(drop=True)
        truth_subset = truth_subset.sort_values('report_id').reset_index(drop=True)
        
        # Initialize KFold
        kf = KFold(n_splits=k, shuffle=True, random_state=42)
        
        # Store results for each fold
        fold_results = {
            'precision': [],
            'recall': [],
            'f1': [],
            'exact_accuracy': []
        }
        
        fold_details = []
        
        # Perform K-fold validation
        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(reports_subset)):
            print(f"Running fold {fold_idx + 1}/{k}...")
            
            # Split data
            test_reports = reports_subset.iloc[test_idx]
            test_truth = truth_subset.iloc[test_idx]
            
            # Initialize extractor and preprocessor
            extractor = self.extractor_class()
            preprocessor = self.preprocessor_class()
            
            # Process test reports
            predictions = []
            for _, row in test_reports.iterrows():
                text = row['text']
                report_id = row['report_id']
                
                # Preprocess and extract
                cleaned_text = preprocessor.clean_text(text)
                extracted = extractor.extract_all(cleaned_text, report_id)
                predictions.append(extracted)
            
            predictions_df = pd.DataFrame(predictions)
            
            # Calculate metrics for this fold
            fold_metrics = self.metrics_calculator.evaluate_extraction_accuracy(
                predictions_df, test_truth
            )
            
            # Store overall metrics
            overall = fold_metrics.get('overall', {})
            fold_results['precision'].append(overall.get('avg_precision', 0.0))
            fold_results['recall'].append(overall.get('avg_recall', 0.0))
            fold_results['f1'].append(overall.get('avg_f1', 0.0))
            
            # Calculate average exact accuracy
            exact_accuracies = [
                metrics.get('exact_accuracy', 0.0) 
                for category, metrics in fold_metrics.items() 
                if category != 'overall' and 'exact_accuracy' in metrics
            ]
            avg_exact_accuracy = np.mean(exact_accuracies) if exact_accuracies else 0.0
            fold_results['exact_accuracy'].append(avg_exact_accuracy)
            
            # Store detailed results
            fold_details.append({
                'fold': fold_idx + 1,
                'test_size': len(test_reports),
                'metrics': fold_metrics,
                'predictions_sample': predictions_df.head(3).to_dict('records')
            })
        
        # Calculate summary statistics
        summary_stats = {}
        for metric, values in fold_results.items():
            summary_stats[metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'values': values
            }
        
        return {
            'summary_stats': summary_stats,
            'fold_details': fold_details,
            'total_reports': len(common_ids),
            'folds': k
        }
    
    def validate_on_test_set(self, test_reports_df: pd.DataFrame, 
                           test_truth_df: pd.DataFrame) -> Dict:
        """
        Validate model on a separate test set.
        
        Args:
            test_reports_df: Test report texts
            test_truth_df: Test ground truth annotations
            
        Returns:
            Validation results
        """
        
        # Initialize model components
        extractor = self.extractor_class()
        preprocessor = self.preprocessor_class()
        
        # Process all test reports
        predictions = []
        for _, row in test_reports_df.iterrows():
            text = row['text']
            report_id = row['report_id']
            
            # Preprocess and extract
            cleaned_text = preprocessor.clean_text(text)
            extracted = extractor.extract_all(cleaned_text, report_id)
            predictions.append(extracted)
        
        predictions_df = pd.DataFrame(predictions)
        
        # Calculate detailed metrics
        metrics = self.metrics_calculator.evaluate_extraction_accuracy(
            predictions_df, test_truth_df
        )
        
        # Generate detailed report
        detailed_report = self.metrics_calculator.generate_detailed_report(
            predictions_df, test_truth_df, "output/test_set_validation_report.json"
        )
        
        return {
            'metrics': metrics,
            'detailed_report': detailed_report,
            'predictions': predictions_df,
            'test_size': len(test_reports_df)
        }
    
    def compare_model_versions(self, reports_df: pd.DataFrame, 
                             ground_truth_df: pd.DataFrame,
                             model_configs: List[Dict]) -> Dict:
        """
        Compare different model configurations.
        
        Args:
            reports_df: Test reports
            ground_truth_df: Ground truth
            model_configs: List of model configurations to compare
            
        Returns:
            Comparison results
        """
        
        comparison_results = {}
        
        for config in model_configs:
            model_name = config.get('name', 'unnamed_model')
            print(f"Evaluating {model_name}...")
            
            # This is a placeholder for when we have multiple model types
            # For now, just validate the rule-based model
            result = self.validate_on_test_set(reports_df, ground_truth_df)
            
            comparison_results[model_name] = {
                'config': config,
                'metrics': result['metrics'],
                'overall_f1': result['metrics']['overall']['avg_f1']
            }
        
        # Rank models by F1 score
        ranked_models = sorted(
            comparison_results.items(), 
            key=lambda x: x[1]['overall_f1'], 
            reverse=True
        )
        
        return {
            'results': comparison_results,
            'ranking': ranked_models,
            'best_model': ranked_models[0] if ranked_models else None
        }
    
    def generate_validation_report(self, validation_results: Dict, 
                                 output_path: str = "output/validation_report.json"):
        """Generate comprehensive validation report."""
        
        report = {
            'validation_summary': {
                'timestamp': pd.Timestamp.now().isoformat(),
                'validation_type': 'k_fold_cross_validation',
                'total_reports': validation_results.get('total_reports', 0),
                'folds': validation_results.get('folds', 0)
            },
            'performance_summary': validation_results.get('summary_stats', {}),
            'fold_details': validation_results.get('fold_details', []),
            'recommendations': self._generate_validation_recommendations(validation_results)
        }
        
        # Save report
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        return report
    
    def _generate_validation_recommendations(self, validation_results: Dict) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []
        
        summary_stats = validation_results.get('summary_stats', {})
        
        # Check F1 score performance
        f1_stats = summary_stats.get('f1', {})
        if f1_stats:
            mean_f1 = f1_stats.get('mean', 0)
            std_f1 = f1_stats.get('std', 0)
            
            if mean_f1 < 0.8:
                recommendations.append(f"Overall F1 score is low ({mean_f1:.3f}). Consider improving extraction patterns or adding ML components.")
            
            if std_f1 > 0.1:
                recommendations.append(f"High variance in F1 scores ({std_f1:.3f}). Model performance is inconsistent across folds.")
        
        # Check precision vs recall balance
        precision_stats = summary_stats.get('precision', {})
        recall_stats = summary_stats.get('recall', {})
        
        if precision_stats and recall_stats:
            mean_precision = precision_stats.get('mean', 0)
            mean_recall = recall_stats.get('mean', 0)
            
            if abs(mean_precision - mean_recall) > 0.15:
                if mean_precision > mean_recall:
                    recommendations.append("Precision is much higher than recall. Consider adding more pattern variations to improve recall.")
                else:
                    recommendations.append("Recall is much higher than precision. Consider refining patterns to reduce false positives.")
        
        # Check exact accuracy
        exact_stats = summary_stats.get('exact_accuracy', {})
        if exact_stats:
            mean_exact = exact_stats.get('mean', 0)
            if mean_exact < 0.7:
                recommendations.append(f"Exact match accuracy is low ({mean_exact:.3f}). Review extraction patterns for accuracy.")
        
        return recommendations

def create_sample_validation_set():
    """Create sample validation dataset for testing."""
    
    # Sample reports for validation
    validation_reports = [
        {
            'report_id': 'val_001',
            'text': '''Patient mit akutem Schlaganfall. Allgemeinanästhesie für Intervention.
                      Beginn: 08:30 Uhr. rtPA bereits verabreicht. Trevo Stentretriever eingesetzt.
                      Mechanische Thrombektomie durchgeführt. TICI 3 Rekanalisierung erreicht.
                      Keine Komplikationen beobachtet.'''
        },
        {
            'report_id': 'val_002', 
            'text': '''67-jähriger Patient, Sedierung für Prozedur. Start um 09:45.
                      SOFIA Aspiration system used. Heparin als Antikoagulation.
                      Leichte Blutung nach Intervention. TICI 2b Ergebnis.'''
        },
        {
            'report_id': 'val_003',
            'text': '''Lokalanästhesie für den Eingriff. Interventionsbeginn: 14:15 Uhr.
                      Solitaire device für mechanische Thrombektomie. Urokinase verabreicht.
                      Perforation der Arterie aufgetreten. TICI 1 erreicht.'''
        },
        {
            'report_id': 'val_004',
            'text': '''Patient in Vollnarkose. Penumbra System verwendet. Beginn: 11:20.
                      Aspiration thrombectomy performed. Tenecteplase given.
                      Erfolgreiche Rekanalisierung. TICI 2c Ergebnis. Kein Hämatom.'''
        },
        {
            'report_id': 'val_005',
            'text': '''Sedierung für Intervention. Catch Mini device eingesetzt.
                      Start der Prozedur: 13:45 Uhr. Alteplase thrombolysis.
                      Embotrap als backup. Minimale Blutung. TICI 3 erreicht.'''
        }
    ]
    
    # Corresponding ground truth annotations
    ground_truth = [
        {
            'report_id': 'val_001',
            'anesthesia': 'allgemeinanästhesie',
            'medication': 'rtpa',
            'device': 'trevo',
            'treatment_method': 'thrombektomie',
            'tici_score': 'tici 3',
            'times': '08:30',
            'complications': None
        },
        {
            'report_id': 'val_002',
            'anesthesia': 'sedierung',
            'medication': 'heparin',
            'device': 'sofia',
            'treatment_method': 'aspiration',
            'tici_score': 'tici 2b',
            'times': '09:45',
            'complications': 'blutung'
        },
        {
            'report_id': 'val_003',
            'anesthesia': 'lokalanästhesie',
            'medication': 'urokinase',
            'device': 'solitaire',
            'treatment_method': 'thrombektomie',
            'tici_score': 'tici 1',
            'times': '14:15',
            'complications': 'perforation'
        },
        {
            'report_id': 'val_004',
            'anesthesia': 'vollnarkose',
            'medication': 'tenecteplase',
            'device': 'penumbra',
            'treatment_method': 'aspiration',
            'tici_score': 'tici 2c',
            'times': '11:20',
            'complications': None
        },
        {
            'report_id': 'val_005',
            'anesthesia': 'sedierung',
            'medication': 'alteplase',
            'device': 'catch mini',
            'treatment_method': None,
            'tici_score': 'tici 3',
            'times': '13:45',
            'complications': 'blutung'
        }
    ]
    
    # Create DataFrames
    reports_df = pd.DataFrame(validation_reports)
    truth_df = pd.DataFrame(ground_truth)
    
    # Save to files
    Path("data/validation").mkdir(parents=True, exist_ok=True)
    reports_df.to_csv("data/validation/validation_reports.csv", index=False)
    truth_df.to_csv("data/validation/validation_ground_truth.csv", index=False)
    
    print("✅ Sample validation dataset created:")
    print(f"  📄 Reports: data/validation/validation_reports.csv ({len(reports_df)} reports)")
    print(f"  📊 Ground Truth: data/validation/validation_ground_truth.csv ({len(truth_df)} annotations)")
    
    return reports_df, truth_df

def run_validation_example():
    """Run a complete validation example."""
    
    print("🧪 Stroke Extraction Model Validation")
    print("=" * 50)
    
    # Create sample validation set
    reports_df, truth_df = create_sample_validation_set()
    
    # Initialize validator
    validator = ModelValidator()
    
    # Run K-fold cross-validation
    print("\n📊 Running 3-fold cross-validation...")
    cv_results = validator.k_fold_validation(reports_df, truth_df, k=3)
    
    # Print results
    print("\n📈 Cross-Validation Results:")
    summary = cv_results['summary_stats']
    
    for metric, stats in summary.items():
        print(f"  {metric.upper()}:")
        print(f"    Mean: {stats['mean']:.3f} ± {stats['std']:.3f}")
        print(f"    Range: {stats['min']:.3f} - {stats['max']:.3f}")
    
    # Generate validation report
    validator.generate_validation_report(cv_results)
    print(f"\n📄 Detailed validation report: output/validation_report.json")
    
    # Test set validation
    print(f"\n🎯 Test Set Validation:")
    test_results = validator.validate_on_test_set(reports_df, truth_df)
    
    overall_metrics = test_results['metrics']['overall']
    print(f"  Overall F1: {overall_metrics['avg_f1']:.3f}")
    print(f"  Overall Precision: {overall_metrics['avg_precision']:.3f}")
    print(f"  Overall Recall: {overall_metrics['avg_recall']:.3f}")
    
    return cv_results, test_results

if __name__ == "__main__":
    # Run validation example
    cv_results, test_results = run_validation_example()
    
    print("\n🎉 Validation complete!")
    print("📊 Check output/ directory for detailed reports")