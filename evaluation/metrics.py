import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

class AccuracyMetrics:
    """Calculate accuracy metrics for stroke report extraction."""
    
    def __init__(self):
        self.categories = [
            'anesthesia', 'medication', 'device', 'treatment_method',
            'tici_score', 'times', 'complications'
        ]
    
    def calculate_binary_metrics(self, y_true: List[bool], y_pred: List[bool]) -> Dict[str, float]:
        """Calculate precision, recall, F1 for binary classification."""
        if len(y_true) == 0 or len(y_pred) == 0:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        return {
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1)
        }
    
    def evaluate_extraction_accuracy(self, predictions: pd.DataFrame, 
                                   ground_truth: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """
        Evaluate extraction accuracy against ground truth.
        
        Args:
            predictions: DataFrame with extracted results
            ground_truth: DataFrame with manually annotated ground truth
            
        Returns:
            Dictionary with metrics per category
        """
        results = {}
        
        # Ensure both DataFrames have the same reports
        common_ids = set(predictions['report_id']) & set(ground_truth['report_id'])
        if not common_ids:
            raise ValueError("No common report IDs found between predictions and ground truth")
        
        pred_subset = predictions[predictions['report_id'].isin(common_ids)].copy()
        truth_subset = ground_truth[ground_truth['report_id'].isin(common_ids)].copy()
        
        # Sort by report_id for consistent comparison
        pred_subset = pred_subset.sort_values('report_id').reset_index(drop=True)
        truth_subset = truth_subset.sort_values('report_id').reset_index(drop=True)
        
        # Calculate metrics for each category
        for category in self.categories:
            if category not in pred_subset.columns or category not in truth_subset.columns:
                results[category] = {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'note': 'Category missing'}
                continue
            
            # Convert to binary (present/absent)
            y_pred = pred_subset[category].notna().tolist()
            y_true = truth_subset[category].notna().tolist()
            
            # Calculate exact match accuracy for non-null values
            exact_matches = []
            for pred_val, true_val in zip(pred_subset[category], truth_subset[category]):
                if pd.isna(true_val) and pd.isna(pred_val):
                    exact_matches.append(True)  # Both are null
                elif pd.isna(true_val) or pd.isna(pred_val):
                    exact_matches.append(False)  # One is null, other isn't
                else:
                    # Both have values - check if they match (case insensitive)
                    exact_matches.append(str(pred_val).lower().strip() == str(true_val).lower().strip())
            
            # Binary presence/absence metrics
            binary_metrics = self.calculate_binary_metrics(y_true, y_pred)
            
            # Exact match accuracy
            exact_accuracy = sum(exact_matches) / len(exact_matches) if exact_matches else 0.0
            
            results[category] = {
                **binary_metrics,
                'exact_accuracy': exact_accuracy,
                'total_samples': len(exact_matches)
            }
        
        # Calculate overall metrics
        all_precisions = [r['precision'] for r in results.values() if 'precision' in r]
        all_recalls = [r['recall'] for r in results.values() if 'recall' in r]
        all_f1s = [r['f1'] for r in results.values() if 'f1' in r]
        
        results['overall'] = {
            'avg_precision': np.mean(all_precisions) if all_precisions else 0.0,
            'avg_recall': np.mean(all_recalls) if all_recalls else 0.0,
            'avg_f1': np.mean(all_f1s) if all_f1s else 0.0,
            'categories_evaluated': len(all_precisions)
        }
        
        return results
    
    def generate_confusion_matrix(self, predictions: pd.DataFrame, 
                                ground_truth: pd.DataFrame, 
                                category: str) -> np.ndarray:
        """Generate confusion matrix for a specific category."""
        if category not in predictions.columns or category not in ground_truth.columns:
            return np.array([[0, 0], [0, 0]])
        
        # Find common reports
        common_ids = set(predictions['report_id']) & set(ground_truth['report_id'])
        pred_subset = predictions[predictions['report_id'].isin(common_ids)].copy()
        truth_subset = ground_truth[ground_truth['report_id'].isin(common_ids)].copy()
        
        # Sort by report_id
        pred_subset = pred_subset.sort_values('report_id')
        truth_subset = truth_subset.sort_values('report_id')
        
        # Convert to binary
        y_pred = pred_subset[category].notna().astype(int).tolist()
        y_true = truth_subset[category].notna().astype(int).tolist()
        
        return confusion_matrix(y_true, y_pred, labels=[0, 1])
    
    def confidence_score_analysis(self, predictions: pd.DataFrame, 
                                ground_truth: pd.DataFrame) -> Dict[str, any]:
        """Analyze confidence score calibration."""
        # This would be implemented when confidence scores are available
        # For now, return placeholder
        return {
            'note': 'Confidence score analysis not yet implemented',
            'avg_confidence': 0.85,  # Placeholder based on rule-based approach
            'confidence_bins': [0.8, 0.85, 0.9, 0.95, 1.0],
            'accuracy_by_confidence': [0.75, 0.80, 0.85, 0.90, 0.95]
        }
    
    def generate_detailed_report(self, predictions: pd.DataFrame, 
                               ground_truth: pd.DataFrame, 
                               output_path: str = "output/accuracy_report.json") -> Dict:
        """Generate comprehensive accuracy report."""
        
        # Calculate main metrics
        accuracy_metrics = self.evaluate_extraction_accuracy(predictions, ground_truth)
        
        # Generate confusion matrices
        confusion_matrices = {}
        for category in self.categories:
            if category in predictions.columns and category in ground_truth.columns:
                cm = self.generate_confusion_matrix(predictions, ground_truth, category)
                confusion_matrices[category] = cm.tolist()  # Convert to list for JSON serialization
        
        # Confidence analysis
        confidence_analysis = self.confidence_score_analysis(predictions, ground_truth)
        
        # Error analysis
        error_analysis = self.analyze_common_errors(predictions, ground_truth)
        
        # Compile full report
        report = {
            'evaluation_summary': {
                'total_reports': len(set(predictions['report_id']) & set(ground_truth['report_id'])),
                'categories_evaluated': len(self.categories),
                'timestamp': pd.Timestamp.now().isoformat()
            },
            'accuracy_metrics': accuracy_metrics,
            'confusion_matrices': confusion_matrices,
            'confidence_analysis': confidence_analysis,
            'error_analysis': error_analysis,
            'recommendations': self.generate_recommendations(accuracy_metrics)
        }
        
        # Save report
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        return report
    
    def analyze_common_errors(self, predictions: pd.DataFrame, 
                            ground_truth: pd.DataFrame) -> Dict[str, List[str]]:
        """Analyze common extraction errors."""
        errors = {
            'false_positives': [],
            'false_negatives': [],
            'misclassifications': []
        }
        
        # Find common reports
        common_ids = set(predictions['report_id']) & set(ground_truth['report_id'])
        
        for category in self.categories:
            if category not in predictions.columns or category not in ground_truth.columns:
                continue
                
            pred_subset = predictions[predictions['report_id'].isin(common_ids)].copy()
            truth_subset = ground_truth[ground_truth['report_id'].isin(common_ids)].copy()
            
            # Sort by report_id
            pred_subset = pred_subset.sort_values('report_id')
            truth_subset = truth_subset.sort_values('report_id')
            
            for idx, (pred_val, true_val) in enumerate(zip(pred_subset[category], truth_subset[category])):
                report_id = pred_subset.iloc[idx]['report_id']
                
                # False positive: predicted something, but ground truth is null
                if pd.notna(pred_val) and pd.isna(true_val):
                    errors['false_positives'].append(f"{category}:{report_id} - predicted '{pred_val}' but should be null")
                
                # False negative: didn't predict, but ground truth has value
                elif pd.isna(pred_val) and pd.notna(true_val):
                    errors['false_negatives'].append(f"{category}:{report_id} - missed '{true_val}'")
                
                # Misclassification: both have values but they don't match
                elif pd.notna(pred_val) and pd.notna(true_val) and str(pred_val).lower() != str(true_val).lower():
                    errors['misclassifications'].append(f"{category}:{report_id} - predicted '{pred_val}' but should be '{true_val}'")
        
        return errors
    
    def generate_recommendations(self, metrics: Dict[str, Dict[str, float]]) -> List[str]:
        """Generate improvement recommendations based on metrics."""
        recommendations = []
        
        for category, category_metrics in metrics.items():
            if category == 'overall':
                continue
                
            if 'precision' in category_metrics and 'recall' in category_metrics:
                precision = category_metrics['precision']
                recall = category_metrics['recall']
                
                if precision < 0.8:
                    recommendations.append(f"Improve {category} precision ({precision:.2f}) by refining extraction patterns to reduce false positives")
                
                if recall < 0.8:
                    recommendations.append(f"Improve {category} recall ({recall:.2f}) by adding more pattern variations to catch missed cases")
                
                if abs(precision - recall) > 0.2:
                    recommendations.append(f"Balance {category} precision-recall trade-off (P={precision:.2f}, R={recall:.2f})")
        
        # Overall recommendations
        overall = metrics.get('overall', {})
        if overall.get('avg_f1', 0) < 0.8:
            recommendations.append("Consider implementing ML-based extraction to improve overall F1 score")
        
        return recommendations

def plot_confidence_distribution(predictions_file: str = "output/extracted_keywords.csv",
                               output_dir: str = "output/"):
    """Plot confidence score distribution (placeholder for future implementation)."""
    
    # Create sample confidence distribution plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Sample data - replace with actual confidence scores when available
    confidence_scores = np.random.beta(8, 2, 1000)  # Beta distribution skewed towards high confidence
    
    ax.hist(confidence_scores, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    ax.set_xlabel('Confidence Score')
    ax.set_ylabel('Frequency')
    ax.set_title('Extraction Confidence Score Distribution')
    ax.grid(True, alpha=0.3)
    
    # Add statistics
    mean_conf = np.mean(confidence_scores)
    ax.axvline(mean_conf, color='red', linestyle='--', label=f'Mean: {mean_conf:.3f}')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/confidence_distribution.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Confidence distribution plot saved to {output_dir}/confidence_distribution.png")

def calculate_category_wise_metrics(predictions_file: str, ground_truth_file: str) -> Dict:
    """Calculate metrics for each extraction category."""
    
    # Load data
    predictions = pd.read_csv(predictions_file)
    ground_truth = pd.read_csv(ground_truth_file)
    
    # Initialize metrics calculator
    metrics_calc = AccuracyMetrics()
    
    # Calculate metrics
    results = metrics_calc.evaluate_extraction_accuracy(predictions, ground_truth)
    
    # Print results
    print("=== STROKE EXTRACTION ACCURACY METRICS ===")
    print()
    
    for category, metrics in results.items():
        if category == 'overall':
            print(f"OVERALL PERFORMANCE:")
            print(f"  Average Precision: {metrics['avg_precision']:.3f}")
            print(f"  Average Recall: {metrics['avg_recall']:.3f}")
            print(f"  Average F1: {metrics['avg_f1']:.3f}")
            print(f"  Categories Evaluated: {metrics['categories_evaluated']}")
        else:
            print(f"{category.upper()}:")
            print(f"  Precision: {metrics.get('precision', 0):.3f}")
            print(f"  Recall: {metrics.get('recall', 0):.3f}")
            print(f"  F1: {metrics.get('f1', 0):.3f}")
            print(f"  Exact Accuracy: {metrics.get('exact_accuracy', 0):.3f}")
            print(f"  Samples: {metrics.get('total_samples', 0)}")
        print()
    
    return results

if __name__ == "__main__":
    # Example usage
    print("🧪 Stroke Extraction Accuracy Metrics")
    print("=" * 50)
    
    # Check if files exist
    pred_file = "output/extracted_keywords.csv"
    truth_file = "evaluation/gold_standard.csv"
    
    if Path(pred_file).exists() and Path(truth_file).exists():
        results = calculate_category_wise_metrics(pred_file, truth_file)
        
        # Generate detailed report
        predictions = pd.read_csv(pred_file)
        ground_truth = pd.read_csv(truth_file)
        
        metrics_calc = AccuracyMetrics()
        report = metrics_calc.generate_detailed_report(predictions, ground_truth)
        
        print(f"\n📊 Detailed report saved to: output/accuracy_report.json")
        print(f"📈 Check recommendations: {len(report['recommendations'])} suggestions")
        
    else:
        print(f"❌ Required files not found:")
        print(f"  Predictions: {pred_file} - {'✓' if Path(pred_file).exists() else '✗'}")
        print(f"  Ground Truth: {truth_file} - {'✓' if Path(truth_file).exists() else '✗'}")
        print(f"\n💡 Run 'python main.py' first to generate predictions")
        print(f"💡 Create gold_standard.csv with manual annotations")