#!/usr/bin/env python3
"""
Training script for stroke radiology report extraction models.

This script provides a comprehensive training pipeline for both rule-based
pattern improvement and ML model training.
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import json
import sys
import os
from typing import Dict, List, Optional

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from extractor.keyword_rules import KeywordExtractor
from extractor.preprocessing import TextPreprocessor
from extractor.ml_model import StrokeMLExtractor, create_sample_training_data
from evaluation.metrics import AccuracyMetrics
from evaluation.validation import ModelValidator

class TrainingPipeline:
    """Main training pipeline for stroke extraction models."""
    
    def __init__(self, config: Dict = None):
        self.config = config or self._get_default_config()
        self.preprocessor = TextPreprocessor()
        self.rule_extractor = KeywordExtractor()
        self.ml_extractor = StrokeMLExtractor()
        self.metrics_calculator = AccuracyMetrics()
        self.validator = ModelValidator()
        
    def _get_default_config(self) -> Dict:
        """Get default training configuration."""
        return {
            'data': {
                'training_reports': 'data/training/training_reports.csv',
                'training_ground_truth': 'data/training/training_ground_truth.csv',
                'validation_reports': 'data/validation/validation_reports.csv',
                'validation_ground_truth': 'data/validation/validation_ground_truth.csv',
                'output_dir': 'output/',
                'models_dir': 'models/'
            },
            'training': {
                'test_size': 0.2,
                'validation_folds': 5,
                'random_state': 42
            },
            'models': {
                'rule_based': True,
                'ml_models': True,
                'ensemble': False  # Future feature
            },
            'evaluation': {
                'generate_reports': True,
                'plot_metrics': True,
                'detailed_analysis': True
            }
        }
    
    def load_training_data(self) -> tuple:
        """Load training and validation datasets."""
        
        data_config = self.config['data']
        
        # Load training data
        try:
            training_reports = pd.read_csv(data_config['training_reports'])
            training_ground_truth = pd.read_csv(data_config['training_ground_truth'])
            print(f"✅ Loaded training data: {len(training_reports)} reports")
        except FileNotFoundError:
            print("📄 Training data not found, creating sample dataset...")
            training_reports, training_ground_truth = create_sample_training_data()
        
        # Load validation data
        try:
            validation_reports = pd.read_csv(data_config['validation_reports'])
            validation_ground_truth = pd.read_csv(data_config['validation_ground_truth'])
            print(f"✅ Loaded validation data: {len(validation_reports)} reports")
        except FileNotFoundError:
            print("📄 Validation data not found, using training data split...")
            # Split training data for validation
            split_idx = int(len(training_reports) * 0.8)
            validation_reports = training_reports.iloc[split_idx:].copy().reset_index(drop=True)
            validation_ground_truth = training_ground_truth.iloc[split_idx:].copy().reset_index(drop=True)
            training_reports = training_reports.iloc[:split_idx].copy().reset_index(drop=True)
            training_ground_truth = training_ground_truth.iloc[:split_idx].copy().reset_index(drop=True)
        
        return training_reports, training_ground_truth, validation_reports, validation_ground_truth
    
    def preprocess_data(self, reports_df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess report texts for training."""
        
        print("🔄 Preprocessing training data...")
        
        processed_reports = reports_df.copy()
        processed_texts = []
        
        for _, row in reports_df.iterrows():
            text = row['text']
            
            # Apply preprocessing
            cleaned_text = self.preprocessor.preprocess_for_training(
                text, 
                remove_patient_info=True,
                normalize_medical_terms=True
            )
            
            processed_texts.append(cleaned_text)
        
        processed_reports['cleaned_text'] = processed_texts
        processed_reports['original_text'] = processed_reports['text']
        processed_reports['text'] = processed_texts
        
        print(f"✅ Preprocessed {len(processed_reports)} reports")
        return processed_reports
    
    def evaluate_rule_based_model(self, reports_df: pd.DataFrame, 
                                 ground_truth_df: pd.DataFrame) -> Dict:
        """Evaluate rule-based extraction performance."""
        
        print("📊 Evaluating rule-based model...")
        
        # Extract using rule-based model
        predictions = []
        for _, row in reports_df.iterrows():
            text = row['text']
            report_id = row['report_id']
            
            # Preprocess and extract
            cleaned_text = self.preprocessor.clean_text(text)
            extracted = self.rule_extractor.extract_all(cleaned_text, report_id)
            predictions.append(extracted)
        
        predictions_df = pd.DataFrame(predictions)
        
        # Calculate metrics
        metrics = self.metrics_calculator.evaluate_extraction_accuracy(
            predictions_df, ground_truth_df
        )
        
        # Run cross-validation
        cv_results = self.validator.k_fold_validation(
            reports_df, ground_truth_df, k=self.config['training']['validation_folds']
        )
        
        return {
            'model_type': 'rule_based',
            'metrics': metrics,
            'cross_validation': cv_results,
            'predictions': predictions_df
        }
    
    def train_ml_models(self, training_reports: pd.DataFrame, 
                       training_ground_truth: pd.DataFrame,
                       validation_reports: pd.DataFrame,
                       validation_ground_truth: pd.DataFrame) -> Dict:
        """Train machine learning models."""
        
        print("🤖 Training ML models...")
        
        # Train models
        training_results = self.ml_extractor.train_all_models(
            training_reports, training_ground_truth
        )
        
        # Evaluate on validation set
        validation_predictions = []
        for _, row in validation_reports.iterrows():
            text = row['text']
            report_id = row['report_id']
            
            extracted = self.ml_extractor.extract_with_ml(text, report_id)
            validation_predictions.append(extracted)
        
        predictions_df = pd.DataFrame(validation_predictions)
        
        # Calculate validation metrics
        validation_metrics = self.metrics_calculator.evaluate_extraction_accuracy(
            predictions_df, validation_ground_truth
        )
        
        # Save models
        saved_models = self.ml_extractor.save_models(self.config['data']['models_dir'])
        
        return {
            'model_type': 'ml_based',
            'training_results': training_results,
            'validation_metrics': validation_metrics,
            'saved_models': saved_models,
            'predictions': predictions_df
        }
    
    def compare_models(self, rule_results: Dict, ml_results: Dict) -> Dict:
        """Compare rule-based and ML model performances."""
        
        print("⚖️ Comparing model performances...")
        
        comparison = {
            'rule_based': {
                'overall_f1': rule_results['metrics']['overall']['avg_f1'],
                'overall_precision': rule_results['metrics']['overall']['avg_precision'],
                'overall_recall': rule_results['metrics']['overall']['avg_recall'],
                'cv_f1_mean': rule_results['cross_validation']['summary_stats']['f1']['mean'],
                'cv_f1_std': rule_results['cross_validation']['summary_stats']['f1']['std']
            },
            'ml_based': {
                'overall_f1': ml_results['validation_metrics']['overall']['avg_f1'],
                'overall_precision': ml_results['validation_metrics']['overall']['avg_precision'],
                'overall_recall': ml_results['validation_metrics']['overall']['avg_recall'],
                'successful_models': ml_results['training_results']['successful_models'],
                'total_categories': ml_results['training_results']['total_categories']
            }
        }
        
        # Determine best model
        rule_f1 = comparison['rule_based']['overall_f1']
        ml_f1 = comparison['ml_based']['overall_f1']
        
        comparison['best_model'] = 'rule_based' if rule_f1 > ml_f1 else 'ml_based'
        comparison['f1_difference'] = abs(rule_f1 - ml_f1)
        
        # Recommendations
        recommendations = []
        if rule_f1 > ml_f1:
            recommendations.append("Rule-based model performs better. Consider refining patterns further.")
        else:
            recommendations.append("ML model shows promise. Consider expanding training data.")
        
        if comparison['f1_difference'] < 0.05:
            recommendations.append("Performance is similar. Consider ensemble approach.")
        
        comparison['recommendations'] = recommendations
        
        return comparison
    
    def generate_training_report(self, rule_results: Dict, ml_results: Dict, 
                               comparison: Dict) -> Dict:
        """Generate comprehensive training report."""
        
        report = {
            'training_summary': {
                'timestamp': pd.Timestamp.now().isoformat(),
                'config': self.config,
                'data_summary': {
                    'training_reports': len(rule_results['predictions']),
                    'validation_reports': len(ml_results['predictions']) if ml_results else 0
                }
            },
            'rule_based_results': rule_results,
            'ml_results': ml_results if ml_results else None,
            'model_comparison': comparison,
            'next_steps': self._generate_next_steps(comparison)
        }
        
        # Save report
        output_dir = Path(self.config['data']['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        report_path = output_dir / 'training_report.json'
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"📄 Training report saved: {report_path}")
        
        return report
    
    def _generate_next_steps(self, comparison: Dict) -> List[str]:
        """Generate next steps based on training results."""
        
        next_steps = []
        
        best_model = comparison.get('best_model', 'unknown')
        f1_diff = comparison.get('f1_difference', 0)
        
        if best_model == 'rule_based':
            next_steps.append("Focus on improving rule-based patterns")
            next_steps.append("Collect more diverse training data for ML models")
        elif best_model == 'ml_based':
            next_steps.append("Expand ML training dataset")
            next_steps.append("Fine-tune ML model hyperparameters")
        
        if f1_diff < 0.05:
            next_steps.append("Implement ensemble approach combining both models")
        
        # General recommendations
        next_steps.extend([
            "Collect real-world radiology reports for training",
            "Implement confidence calibration",
            "Create production deployment pipeline",
            "Set up continuous model monitoring"
        ])
        
        return next_steps
    
    def run_full_training_pipeline(self):
        """Run the complete training pipeline."""
        
        print("🚀 Starting Stroke Extraction Model Training Pipeline")
        print("=" * 60)
        
        try:
            # Load data
            training_reports, training_ground_truth, validation_reports, validation_ground_truth = self.load_training_data()
            
            # Preprocess data
            training_reports = self.preprocess_data(training_reports)
            validation_reports = self.preprocess_data(validation_reports)
            
            results = {}
            
            # Train and evaluate rule-based model
            if self.config['models']['rule_based']:
                rule_results = self.evaluate_rule_based_model(validation_reports, validation_ground_truth)
                results['rule_based'] = rule_results
                
                print(f"📊 Rule-based F1: {rule_results['metrics']['overall']['avg_f1']:.3f}")
            
            # Train ML models
            ml_results = None
            if self.config['models']['ml_models']:
                try:
                    ml_results = self.train_ml_models(
                        training_reports, training_ground_truth,
                        validation_reports, validation_ground_truth
                    )
                    results['ml_based'] = ml_results
                    
                    print(f"🤖 ML-based F1: {ml_results['validation_metrics']['overall']['avg_f1']:.3f}")
                except Exception as e:
                    print(f"⚠️ ML training failed: {e}")
                    print("Continuing with rule-based model only...")
            
            # Compare models
            if 'rule_based' in results and ml_results:
                comparison = self.compare_models(results['rule_based'], ml_results)
                results['comparison'] = comparison
                
                print(f"🏆 Best model: {comparison['best_model']}")
            else:
                comparison = {'best_model': 'rule_based', 'note': 'Only rule-based model available'}
                results['comparison'] = comparison
            
            # Generate report
            report = self.generate_training_report(
                results.get('rule_based'), 
                results.get('ml_based'),
                comparison
            )
            
            print("\n✅ Training pipeline completed successfully!")
            print(f"📊 Check {self.config['data']['output_dir']} for detailed results")
            
            return results
            
        except Exception as e:
            print(f"❌ Training pipeline failed: {e}")
            raise

def main():
    """Main training script entry point."""
    
    parser = argparse.ArgumentParser(description="Train stroke extraction models")
    parser.add_argument('--data', type=str, help='Path to training data directory')
    parser.add_argument('--validation', type=str, help='Path to validation data directory')
    parser.add_argument('--output', type=str, default='output/', help='Output directory')
    parser.add_argument('--models-dir', type=str, default='models/', help='Models directory')
    parser.add_argument('--rule-only', action='store_true', help='Train only rule-based model')
    parser.add_argument('--ml-only', action='store_true', help='Train only ML models')
    parser.add_argument('--config', type=str, help='Path to configuration JSON file')
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config and Path(args.config).exists():
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        config = {}
    
    # Override config with command line arguments
    if args.data:
        config.setdefault('data', {})['training_reports'] = f"{args.data}/training_reports.csv"
        config['data']['training_ground_truth'] = f"{args.data}/training_ground_truth.csv"
    
    if args.validation:
        config.setdefault('data', {})['validation_reports'] = f"{args.validation}/validation_reports.csv"
        config['data']['validation_ground_truth'] = f"{args.validation}/validation_ground_truth.csv"
    
    if args.output:
        config.setdefault('data', {})['output_dir'] = args.output
    
    if args.models_dir:
        config.setdefault('data', {})['models_dir'] = args.models_dir
    
    if args.rule_only:
        config.setdefault('models', {})['ml_models'] = False
    
    if args.ml_only:
        config.setdefault('models', {})['rule_based'] = False
    
    # Initialize and run training pipeline
    pipeline = TrainingPipeline(config)
    results = pipeline.run_full_training_pipeline()
    
    return results

if __name__ == "__main__":
    # Example usage when run directly
    if len(sys.argv) == 1:
        print("🧪 Running training pipeline with default settings...")
        pipeline = TrainingPipeline()
        results = pipeline.run_full_training_pipeline()
    else:
        results = main()