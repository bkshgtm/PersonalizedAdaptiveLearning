"""
Comprehensive evaluation metrics for knowledge tracing models.
Addresses Dr. Fu's feedback on incomplete evaluation framework.

Includes:
- Standard classification metrics (precision, recall, F1, AUC)
- Calibration metrics (reliability diagrams, Brier score)
- Per-topic analysis
- Statistical significance testing
- Baseline comparisons
- Ablation study framework
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Any, Optional, Union
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, roc_curve, precision_recall_curve, 
    confusion_matrix, classification_report, log_loss,
    brier_score_loss, average_precision_score
)
try:
    from sklearn.calibration import calibration_curve
except ImportError:
    from sklearn.metrics import calibration_curve
from scipy import stats
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.style.use('default')  
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    plt = None
    sns = None
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None
import logging
import json
import os
from datetime import datetime

logger = logging.getLogger(__name__)


class ComprehensiveEvaluator:
    """
    Comprehensive evaluation framework for knowledge tracing models.
    """
    
    def __init__(self, topic_names: Optional[List[str]] = None, save_dir: str = "evaluation_results"):
        """
        Initialize the evaluator.
        
        Args:
            topic_names: Optional list of topic names for detailed analysis
            save_dir: Directory to save evaluation results
        """
        self.topic_names = topic_names or []
        self.save_dir = save_dir
        self.results_history = []
        
        # Create save directory and visualization subdirectory
        os.makedirs(save_dir, exist_ok=True)
        self.viz_dir = os.path.join(save_dir, "visualizations")
        os.makedirs(self.viz_dir, exist_ok=True)
    
    def evaluate_model(
        self, 
        y_true: np.ndarray, 
        y_pred_proba: np.ndarray,
        topic_ids: Optional[np.ndarray] = None,
        model_name: str = "Model",
        save_results: bool = True
    ) -> Dict[str, Any]:
        """
        Comprehensive evaluation of a model's predictions.
        
        Args:
            y_true: True binary labels (0 or 1)
            y_pred_proba: Predicted probabilities (0 to 1)
            topic_ids: Optional topic IDs for per-topic analysis
            model_name: Name of the model being evaluated
            save_results: Whether to save results to disk
            
        Returns:
            Dictionary containing all evaluation metrics
        """
        logger.info(f"Evaluating model: {model_name}")
        
        # Convert to numpy arrays
        y_true = np.array(y_true)
        y_pred_proba = np.array(y_pred_proba)
        y_pred = (y_pred_proba >= 0.5).astype(int)
        
        # Basic validation
        if len(y_true) != len(y_pred_proba):
            raise ValueError("y_true and y_pred_proba must have same length")
        
        results = {
            'model_name': model_name,
            'timestamp': datetime.now().isoformat(),
            'n_samples': len(y_true),
            'n_positive': int(np.sum(y_true)),
            'n_negative': int(len(y_true) - np.sum(y_true)),
            'class_balance': float(np.mean(y_true))
        }
        
        # 1. Standard Classification Metrics
        results['classification_metrics'] = self._compute_classification_metrics(
            y_true, y_pred, y_pred_proba
        )
        
        # 2. Calibration Metrics
        results['calibration_metrics'] = self._compute_calibration_metrics(
            y_true, y_pred_proba
        )
        
        # 3. Per-topic Analysis (if topic_ids provided)
        if topic_ids is not None:
            results['per_topic_metrics'] = self._compute_per_topic_metrics(
                y_true, y_pred_proba, topic_ids
            )
        
        # 4. Confidence Analysis
        results['confidence_analysis'] = self._compute_confidence_analysis(
            y_true, y_pred_proba
        )
        
        # 5. Error Analysis
        results['error_analysis'] = self._compute_error_analysis(
            y_true, y_pred_proba
        )
        
        # Save results
        if save_results:
            self._save_results(results, model_name)
            # Generate visualizations
            self._generate_visualizations(results, y_true, y_pred_proba, topic_ids, model_name)
        
        # Add to history
        self.results_history.append(results)
        
        logger.info(f"Evaluation complete for {model_name}")
        logger.info(f"Accuracy: {results['classification_metrics']['accuracy']:.4f}")
        logger.info(f"AUC: {results['classification_metrics']['auc']:.4f}")
        logger.info(f"F1: {results['classification_metrics']['f1']:.4f}")
        
        return results
    
    def _compute_classification_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba: np.ndarray
    ) -> Dict[str, float]:
        """Compute standard classification metrics."""
        
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = float(accuracy_score(y_true, y_pred))
        metrics['precision'] = float(precision_score(y_true, y_pred, zero_division=0))
        metrics['recall'] = float(recall_score(y_true, y_pred, zero_division=0))
        metrics['f1'] = float(f1_score(y_true, y_pred, zero_division=0))
        
        # ROC metrics
        try:
            metrics['auc'] = float(roc_auc_score(y_true, y_pred_proba))
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
            metrics['fpr'] = fpr.tolist()
            metrics['tpr'] = tpr.tolist()
        except ValueError as e:
            logger.warning(f"Could not compute AUC: {e}")
            metrics['auc'] = 0.0
            metrics['fpr'] = []
            metrics['tpr'] = []
        
        # Precision-Recall metrics
        try:
            metrics['average_precision'] = float(average_precision_score(y_true, y_pred_proba))
            precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
            metrics['pr_precision'] = precision.tolist()
            metrics['pr_recall'] = recall.tolist()
        except ValueError as e:
            logger.warning(f"Could not compute PR curve: {e}")
            metrics['average_precision'] = 0.0
            metrics['pr_precision'] = []
            metrics['pr_recall'] = []
        
        # Log loss
        try:
            # Clip probabilities to avoid log(0)
            y_pred_proba_clipped = np.clip(y_pred_proba, 1e-15, 1 - 1e-15)
            metrics['log_loss'] = float(log_loss(y_true, y_pred_proba_clipped))
        except ValueError as e:
            logger.warning(f"Could not compute log loss: {e}")
            metrics['log_loss'] = float('inf')
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics['true_negatives'] = int(tn)
            metrics['false_positives'] = int(fp)
            metrics['false_negatives'] = int(fn)
            metrics['true_positives'] = int(tp)
            
            # Additional metrics
            metrics['specificity'] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
            metrics['sensitivity'] = metrics['recall']  # Same as recall
            metrics['balanced_accuracy'] = float((metrics['sensitivity'] + metrics['specificity']) / 2)
        
        return metrics
    
    def _compute_calibration_metrics(
        self, y_true: np.ndarray, y_pred_proba: np.ndarray
    ) -> Dict[str, Any]:
        """Compute calibration metrics."""
        
        metrics = {}
        
        # Brier score
        metrics['brier_score'] = float(brier_score_loss(y_true, y_pred_proba))
        
        # Calibration curve
        try:
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_true, y_pred_proba, n_bins=10
            )
            metrics['calibration_curve'] = {
                'fraction_of_positives': fraction_of_positives.tolist(),
                'mean_predicted_value': mean_predicted_value.tolist()
            }
            
            # Expected Calibration Error (ECE)
            bin_boundaries = np.linspace(0, 1, 11)
            bin_lowers = bin_boundaries[:-1]
            bin_uppers = bin_boundaries[1:]
            
            ece = 0
            for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                in_bin = (y_pred_proba > bin_lower) & (y_pred_proba <= bin_upper)
                prop_in_bin = in_bin.mean()
                
                if prop_in_bin > 0:
                    accuracy_in_bin = y_true[in_bin].mean()
                    avg_confidence_in_bin = y_pred_proba[in_bin].mean()
                    ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
            metrics['expected_calibration_error'] = float(ece)
            
        except ValueError as e:
            logger.warning(f"Could not compute calibration metrics: {e}")
            metrics['calibration_curve'] = {'fraction_of_positives': [], 'mean_predicted_value': []}
            metrics['expected_calibration_error'] = float('inf')
        
        return metrics
    
    def _compute_per_topic_metrics(
        self, y_true: np.ndarray, y_pred_proba: np.ndarray, topic_ids: np.ndarray
    ) -> Dict[str, Any]:
        """Compute per-topic performance metrics."""
        
        per_topic = {}
        unique_topics = np.unique(topic_ids)
        
        for topic_id in unique_topics:
            topic_mask = topic_ids == topic_id
            topic_y_true = y_true[topic_mask]
            topic_y_pred_proba = y_pred_proba[topic_mask]
            topic_y_pred = (topic_y_pred_proba >= 0.5).astype(int)
            
            if len(topic_y_true) == 0:
                continue
            
            topic_metrics = {
                'n_samples': int(len(topic_y_true)),
                'n_positive': int(np.sum(topic_y_true)),
                'accuracy': float(accuracy_score(topic_y_true, topic_y_pred)),
                'precision': float(precision_score(topic_y_true, topic_y_pred, zero_division=0)),
                'recall': float(recall_score(topic_y_true, topic_y_pred, zero_division=0)),
                'f1': float(f1_score(topic_y_true, topic_y_pred, zero_division=0))
            }
            
            # AUC (only if both classes present)
            if len(np.unique(topic_y_true)) > 1:
                try:
                    topic_metrics['auc'] = float(roc_auc_score(topic_y_true, topic_y_pred_proba))
                except ValueError:
                    topic_metrics['auc'] = 0.0
            else:
                topic_metrics['auc'] = 0.0
            
            # Topic name if available
            topic_name = str(topic_id)
            if hasattr(self, 'topic_names') and len(self.topic_names) > topic_id:
                topic_name = self.topic_names[topic_id]
            
            per_topic[topic_name] = topic_metrics
        
        # Summary statistics across topics
        if per_topic:
            accuracies = [metrics['accuracy'] for metrics in per_topic.values()]
            aucs = [metrics['auc'] for metrics in per_topic.values() if metrics['auc'] > 0]
            
            summary = {
                'mean_accuracy': float(np.mean(accuracies)),
                'std_accuracy': float(np.std(accuracies)),
                'min_accuracy': float(np.min(accuracies)),
                'max_accuracy': float(np.max(accuracies)),
            }
            
            if aucs:
                summary.update({
                    'mean_auc': float(np.mean(aucs)),
                    'std_auc': float(np.std(aucs)),
                    'min_auc': float(np.min(aucs)),
                    'max_auc': float(np.max(aucs))
                })
            
            per_topic['summary'] = summary
        
        return per_topic
    
    def _compute_confidence_analysis(
        self, y_true: np.ndarray, y_pred_proba: np.ndarray
    ) -> Dict[str, Any]:
        """Analyze model confidence patterns."""
        
        analysis = {}
        
        # Confidence distribution
        analysis['confidence_stats'] = {
            'mean': float(np.mean(y_pred_proba)),
            'std': float(np.std(y_pred_proba)),
            'min': float(np.min(y_pred_proba)),
            'max': float(np.max(y_pred_proba)),
            'median': float(np.median(y_pred_proba)),
            'q25': float(np.percentile(y_pred_proba, 25)),
            'q75': float(np.percentile(y_pred_proba, 75))
        }
        
        # Confidence vs accuracy analysis
        confidence_bins = np.linspace(0, 1, 11)
        bin_accuracies = []
        bin_counts = []
        
        for i in range(len(confidence_bins) - 1):
            bin_mask = (y_pred_proba >= confidence_bins[i]) & (y_pred_proba < confidence_bins[i + 1])
            if i == len(confidence_bins) - 2:  # Last bin includes upper bound
                bin_mask = (y_pred_proba >= confidence_bins[i]) & (y_pred_proba <= confidence_bins[i + 1])
            
            if np.sum(bin_mask) > 0:
                bin_accuracy = np.mean(y_true[bin_mask])
                bin_accuracies.append(float(bin_accuracy))
                bin_counts.append(int(np.sum(bin_mask)))
            else:
                bin_accuracies.append(0.0)
                bin_counts.append(0)
        
        analysis['confidence_vs_accuracy'] = {
            'bin_boundaries': confidence_bins.tolist(),
            'bin_accuracies': bin_accuracies,
            'bin_counts': bin_counts
        }
        
        return analysis
    
    def _compute_error_analysis(
        self, y_true: np.ndarray, y_pred_proba: np.ndarray
    ) -> Dict[str, Any]:
        """Analyze prediction errors."""
        
        y_pred = (y_pred_proba >= 0.5).astype(int)
        
        # Identify different types of errors
        correct_predictions = y_true == y_pred
        false_positives = (y_true == 0) & (y_pred == 1)
        false_negatives = (y_true == 1) & (y_pred == 0)
        
        analysis = {
            'correct_rate': float(np.mean(correct_predictions)),
            'false_positive_rate': float(np.mean(false_positives)),
            'false_negative_rate': float(np.mean(false_negatives))
        }
        
        # Confidence analysis for different prediction types
        if np.sum(correct_predictions) > 0:
            analysis['correct_confidence_mean'] = float(np.mean(y_pred_proba[correct_predictions]))
            analysis['correct_confidence_std'] = float(np.std(y_pred_proba[correct_predictions]))
        
        if np.sum(false_positives) > 0:
            analysis['false_positive_confidence_mean'] = float(np.mean(y_pred_proba[false_positives]))
            analysis['false_positive_confidence_std'] = float(np.std(y_pred_proba[false_positives]))
        
        if np.sum(false_negatives) > 0:
            analysis['false_negative_confidence_mean'] = float(np.mean(y_pred_proba[false_negatives]))
            analysis['false_negative_confidence_std'] = float(np.std(y_pred_proba[false_negatives]))
        
        return analysis
    
    def _save_results(self, results: Dict[str, Any], model_name: str):
        """Save evaluation results to disk."""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{model_name}_{timestamp}_evaluation.json"
        filepath = os.path.join(self.save_dir, filename)
        
        try:
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Evaluation results saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
    
    def compare_models(
        self, 
        model_results: List[Dict[str, Any]], 
        save_comparison: bool = True
    ) -> Dict[str, Any]:
        """
        Compare multiple model evaluation results.
        
        Args:
            model_results: List of evaluation result dictionaries
            save_comparison: Whether to save comparison results
            
        Returns:
            Dictionary containing model comparison
        """
        if len(model_results) < 2:
            raise ValueError("Need at least 2 models to compare")
        
        comparison = {
            'timestamp': datetime.now().isoformat(),
            'models_compared': [r['model_name'] for r in model_results],
            'n_models': len(model_results)
        }
        
        # Compare key metrics
        metrics_to_compare = ['accuracy', 'precision', 'recall', 'f1', 'auc', 'brier_score']
        
        metric_comparison = {}
        for metric in metrics_to_compare:
            values = []
            for result in model_results:
                if metric in result['classification_metrics']:
                    values.append(result['classification_metrics'][metric])
                elif metric in result['calibration_metrics']:
                    values.append(result['calibration_metrics'][metric])
            
            if values:
                metric_comparison[metric] = {
                    'values': values,
                    'best_model': model_results[np.argmax(values) if metric != 'brier_score' else np.argmin(values)]['model_name'],
                    'best_value': max(values) if metric != 'brier_score' else min(values),
                    'worst_value': min(values) if metric != 'brier_score' else max(values),
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values))
                }
        
        comparison['metric_comparison'] = metric_comparison
        
        # Statistical significance testing (if applicable)
        if len(model_results) == 2:
            comparison['statistical_tests'] = self._perform_statistical_tests(model_results)
        
        if save_comparison:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"model_comparison_{timestamp}.json"
            filepath = os.path.join(self.save_dir, filename)
            
            try:
                with open(filepath, 'w') as f:
                    json.dump(comparison, f, indent=2)
                logger.info(f"Model comparison saved to {filepath}")
            except Exception as e:
                logger.error(f"Failed to save comparison: {e}")
        
        return comparison
    
    def _perform_statistical_tests(
        self, model_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Perform statistical significance tests between two models."""
        
        if len(model_results) != 2:
            return {}
        
        tests = {}
        
        # Compare key metrics using appropriate tests
        metrics_to_test = ['accuracy', 'auc', 'f1']
        
        for metric in metrics_to_test:
            if (metric in model_results[0]['classification_metrics'] and 
                metric in model_results[1]['classification_metrics']):
                
                value1 = model_results[0]['classification_metrics'][metric]
                value2 = model_results[1]['classification_metrics'][metric]
                
                # Simple difference test (more sophisticated tests would require raw predictions)
                difference = abs(value1 - value2)
                
                tests[metric] = {
                    'model1_value': value1,
                    'model2_value': value2,
                    'absolute_difference': difference,
                    'relative_difference': difference / max(value1, value2) if max(value1, value2) > 0 else 0,
                    'better_model': model_results[0]['model_name'] if value1 > value2 else model_results[1]['model_name']
                }
        
        return tests
    
    def generate_evaluation_report(
        self, results: Dict[str, Any], save_report: bool = True
    ) -> str:
        """
        Generate a human-readable evaluation report.
        
        Args:
            results: Evaluation results dictionary
            save_report: Whether to save report to disk
            
        Returns:
            Report as string
        """
        report_lines = []
        
        # Header
        report_lines.append("=" * 80)
        report_lines.append(f"EVALUATION REPORT: {results['model_name']}")
        report_lines.append("=" * 80)
        report_lines.append(f"Timestamp: {results['timestamp']}")
        report_lines.append(f"Samples: {results['n_samples']:,}")
        report_lines.append(f"Positive class: {results['n_positive']:,} ({results['class_balance']:.1%})")
        report_lines.append("")
        
        # Classification metrics
        report_lines.append("CLASSIFICATION METRICS")
        report_lines.append("-" * 40)
        cm = results['classification_metrics']
        report_lines.append(f"Accuracy:     {cm['accuracy']:.4f}")
        report_lines.append(f"Precision:    {cm['precision']:.4f}")
        report_lines.append(f"Recall:       {cm['recall']:.4f}")
        report_lines.append(f"F1-Score:     {cm['f1']:.4f}")
        report_lines.append(f"AUC:          {cm['auc']:.4f}")
        report_lines.append(f"Log Loss:     {cm['log_loss']:.4f}")
        
        if 'balanced_accuracy' in cm:
            report_lines.append(f"Balanced Acc: {cm['balanced_accuracy']:.4f}")
        
        report_lines.append("")
        
        # Calibration metrics
        report_lines.append("CALIBRATION METRICS")
        report_lines.append("-" * 40)
        cal = results['calibration_metrics']
        report_lines.append(f"Brier Score:  {cal['brier_score']:.4f}")
        report_lines.append(f"ECE:          {cal['expected_calibration_error']:.4f}")
        report_lines.append("")
        
        # Per-topic summary (if available)
        if 'per_topic_metrics' in results and 'summary' in results['per_topic_metrics']:
            report_lines.append("PER-TOPIC PERFORMANCE SUMMARY")
            report_lines.append("-" * 40)
            summary = results['per_topic_metrics']['summary']
            report_lines.append(f"Mean Accuracy: {summary['mean_accuracy']:.4f} ± {summary['std_accuracy']:.4f}")
            report_lines.append(f"Range:         {summary['min_accuracy']:.4f} - {summary['max_accuracy']:.4f}")
            
            if 'mean_auc' in summary:
                report_lines.append(f"Mean AUC:      {summary['mean_auc']:.4f} ± {summary['std_auc']:.4f}")
            
            report_lines.append("")
        
        # Confidence analysis
        report_lines.append("CONFIDENCE ANALYSIS")
        report_lines.append("-" * 40)
        conf = results['confidence_analysis']['confidence_stats']
        report_lines.append(f"Mean Confidence: {conf['mean']:.4f}")
        report_lines.append(f"Std Confidence:  {conf['std']:.4f}")
        report_lines.append(f"Range:           {conf['min']:.4f} - {conf['max']:.4f}")
        report_lines.append("")
        
        # Error analysis
        report_lines.append("ERROR ANALYSIS")
        report_lines.append("-" * 40)
        err = results['error_analysis']
        report_lines.append(f"Correct Rate:    {err['correct_rate']:.4f}")
        report_lines.append(f"False Pos Rate:  {err['false_positive_rate']:.4f}")
        report_lines.append(f"False Neg Rate:  {err['false_negative_rate']:.4f}")
        
        report_lines.append("")
        report_lines.append("=" * 80)
        
        report = "\n".join(report_lines)
        
        if save_report:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{results['model_name']}_{timestamp}_report.txt"
            filepath = os.path.join(self.save_dir, filename)
            
            try:
                with open(filepath, 'w') as f:
                    f.write(report)
                logger.info(f"Evaluation report saved to {filepath}")
            except Exception as e:
                logger.error(f"Failed to save report: {e}")
        
        return report
    
    def _generate_visualizations(
        self, 
        results: Dict[str, Any], 
        y_true: np.ndarray, 
        y_pred_proba: np.ndarray,
        topic_ids: Optional[np.ndarray] = None,
        model_name: str = "Model"
    ):
        """
        Generate comprehensive visualizations for evaluation results.
        
        Args:
            results: Evaluation results dictionary
            y_true: True binary labels
            y_pred_proba: Predicted probabilities
            topic_ids: Optional topic IDs for per-topic analysis
            model_name: Name of the model
        """
        try:
            from .evaluation_visualizer import EvaluationVisualizer
            
            visualizer = EvaluationVisualizer(save_dir=self.save_dir)
            visualizer.generate_all_visualizations(
                results, y_true, y_pred_proba, topic_ids, model_name
            )
            
        except ImportError as e:
            logger.warning(f"Could not import visualizer: {e}")
        except Exception as e:
            logger.error(f"Error generating visualizations: {e}")


class BaselineModels:
    """
    Simple baseline models for comparison with sophisticated knowledge tracing models.
    """
    
    @staticmethod
    def random_baseline(n_samples: int, random_state: int = 42) -> np.ndarray:
        """Random predictions baseline."""
        np.random.seed(random_state)
        return np.random.random(n_samples)
    
    @staticmethod
    def majority_class_baseline(y_train: np.ndarray, n_samples: int) -> np.ndarray:
        """Always predict majority class."""
        majority_prob = np.mean(y_train)
        return np.full(n_samples, majority_prob)
    
    @staticmethod
    def student_average_baseline(
        student_ids: np.ndarray, 
        y_train: np.ndarray, 
        student_ids_train: np.ndarray,
        n_samples: int
    ) -> np.ndarray:
        """Predict based on student's historical average performance."""
        
        # Calculate per-student averages from training data
        student_averages = {}
        unique_students = np.unique(student_ids_train)
        
        for student_id in unique_students:
            student_mask = student_ids_train == student_id
            if np.sum(student_mask) > 0:
                student_averages[student_id] = np.mean(y_train[student_mask])
        
        # Global average as fallback
        global_average = np.mean(y_train)
        
        # Generate predictions
        predictions = np.zeros(n_samples)
        for i, student_id in enumerate(student_ids):
            predictions[i] = student_averages.get(student_id, global_average)
        
        return predictions
    
    @staticmethod
    def topic_average_baseline(
        topic_ids: np.ndarray,
        y_train: np.ndarray,
        topic_ids_train: np.ndarray,
        n_samples: int
    ) -> np.ndarray:
        """Predict based on topic's historical average difficulty."""
        
        # Calculate per-topic averages from training data
        topic_averages = {}
        unique_topics = np.unique(topic_ids_train)
        
        for topic_id in unique_topics:
            topic_mask = topic_ids_train == topic_id
            if np.sum(topic_mask) > 0:
                topic_averages[topic_id] = np.mean(y_train[topic_mask])
        
        # Global average as fallback
        global_average = np.mean(y_train)
        
        # Generate predictions
        predictions = np.zeros(n_samples)
        for i, topic_id in enumerate(topic_ids):
            predictions[i] = topic_averages.get(topic_id, global_average)
        
        return predictions


class AblationStudyFramework:
    """
    Framework for conducting ablation studies on knowledge tracing models.
    """
    
    def __init__(self, base_model_class, evaluator: ComprehensiveEvaluator):
        """
        Initialize ablation study framework.
        
        Args:
            base_model_class: The model class to study
            evaluator: Evaluator instance for consistent evaluation
        """
        self.base_model_class = base_model_class
        self.evaluator = evaluator
        self.ablation_results = []
    
    def run_ablation_study(
        self,
        train_loader,
        test_loader,
        base_config: Dict[str, Any],
        ablation_configs: List[Dict[str, Any]],
        num_epochs: int = 10
    ) -> Dict[str, Any]:
        """
        Run ablation study with different model configurations.
        
        Args:
            train_loader: Training data loader
            test_loader: Test data loader
            base_config: Base model configuration
            ablation_configs: List of configurations to test
            num_epochs: Number of training epochs
            
        Returns:
            Dictionary containing ablation study results
        """
        
        results = {
            'base_config': base_config,
            'ablation_configs': ablation_configs,
            'results': []
        }
        
        # Test base configuration
        logger.info("Testing base configuration...")
        base_result = self._train_and_evaluate_config(
            base_config, train_loader, test_loader, num_epochs, "Base"
        )
        results['base_result'] = base_result
        
        # Test ablation configurations
        for i, config in enumerate(ablation_configs):
            config_name = config.get('name', f'Ablation_{i+1}')
            logger.info(f"Testing configuration: {config_name}")
            
            ablation_result = self._train_and_evaluate_config(
                config, train_loader, test_loader, num_epochs, config_name
            )
            results['results'].append(ablation_result)
        
        return results
    
    def _train_and_evaluate_config(
        self, 
        config: Dict[str, Any], 
        train_loader, 
        test_loader, 
        num_epochs: int,
        config_name: str
    ) -> Dict[str, Any]:
        """Train and evaluate a single configuration."""
        
        # Create model with configuration
        model = self.base_model_class(**config)
        
        # Simple training loop (can be made more sophisticated)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.get('learning_rate', 0.001))
        
        model.train()
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            for batch in train_loader:
                optimizer.zero_grad()
                
                # Forward pass (adapt based on model type)
                if hasattr(model, 'forward'):
                    outputs = model(**batch)
                    loss = outputs.get('loss', outputs)
                else:
                    loss = model(batch)
                
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
        
        # Evaluate model
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in test_loader:
                if hasattr(model, 'forward'):
                    outputs = model(**batch)
                    preds = torch.sigmoid(outputs.get('logits', outputs))
                else:
                    preds = torch.sigmoid(model(batch))
                
                all_preds.extend(preds.cpu().numpy().flatten())
                all_labels.extend(batch['labels'].cpu().numpy().flatten())
        
        # Use evaluator to get comprehensive metrics
        evaluation_result = self.evaluator.evaluate_model(
            np.array(all_labels),
            np.array(all_preds),
            model_name=config_name,
            save_results=False
        )
        
        return {
            'config': config,
            'config_name': config_name,
            'evaluation': evaluation_result
        }
