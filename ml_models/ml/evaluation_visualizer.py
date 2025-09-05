"""
Comprehensive visualization module for evaluation metrics.
Generates charts and plots for all evaluation results.
"""

import numpy as np
import os
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional

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

logger = logging.getLogger(__name__)


class EvaluationVisualizer:
    """
    Comprehensive visualization generator for evaluation results.
    """
    
    def __init__(self, save_dir: str = "evaluation_results"):
        """
        Initialize the visualizer.
        
        Args:
            save_dir: Directory to save visualizations
        """
        self.save_dir = save_dir
        self.viz_dir = os.path.join(save_dir, "visualizations")
        os.makedirs(self.viz_dir, exist_ok=True)
        
        # Set up plotting style
        if PLOTTING_AVAILABLE:
            plt.style.use('default')
            sns.set_palette("husl")
    
    def generate_all_visualizations(
        self, 
        results: Dict[str, Any], 
        y_true: np.ndarray, 
        y_pred_proba: np.ndarray,
        topic_ids: Optional[np.ndarray] = None,
        model_name: str = "Model"
    ):
        """
        Generate all visualizations for evaluation results.
        
        Args:
            results: Evaluation results dictionary
            y_true: True binary labels
            y_pred_proba: Predicted probabilities
            topic_ids: Optional topic IDs for per-topic analysis
            model_name: Name of the model
        """
        if not PLOTTING_AVAILABLE:
            logger.warning("Matplotlib/seaborn not available. Skipping visualizations.")
            return
        
        logger.info(f"Generating visualizations for {model_name}")
        
        # Create model-specific visualization directory
        model_viz_dir = os.path.join(self.viz_dir, model_name.replace(" ", "_"))
        os.makedirs(model_viz_dir, exist_ok=True)
        
        try:
            # 1. Performance Metrics Overview
            self._plot_metrics_overview(results, model_viz_dir, model_name)
            
            # 2. ROC Curve
            self._plot_roc_curve(results, model_viz_dir, model_name)
            
            # 3. Precision-Recall Curve
            self._plot_precision_recall_curve(results, model_viz_dir, model_name)
            
            # 4. Calibration Plot
            self._plot_calibration_curve(results, model_viz_dir, model_name)
            
            # 5. Confusion Matrix
            self._plot_confusion_matrix(results, model_viz_dir, model_name)
            
            # 6. Confidence Distribution
            self._plot_confidence_distribution(y_pred_proba, y_true, model_viz_dir, model_name)
            
            # 7. Confidence vs Accuracy
            self._plot_confidence_vs_accuracy(results, model_viz_dir, model_name)
            
            # 8. Error Analysis
            self._plot_error_analysis(results, y_pred_proba, y_true, model_viz_dir, model_name)
            
            # 9. Per-topic Analysis (if available)
            if topic_ids is not None and 'per_topic_metrics' in results:
                self._plot_per_topic_analysis(results, model_viz_dir, model_name)
            
            logger.info(f"All visualizations saved to {model_viz_dir}")
            
        except Exception as e:
            logger.error(f"Error generating visualizations: {e}")
    
    def _plot_metrics_overview(self, results: Dict[str, Any], save_dir: str, model_name: str):
        """Plot overview of key performance metrics."""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'{model_name} - Performance Metrics Overview', fontsize=16, fontweight='bold')
        
        # Classification metrics
        cm = results['classification_metrics']
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        values = [cm['accuracy'], cm['precision'], cm['recall'], cm['f1']]
        
        bars = ax1.bar(metrics, values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        ax1.set_title('Classification Metrics')
        ax1.set_ylabel('Score')
        ax1.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{value:.3f}', ha='center', va='bottom')
        
        # AUC and Calibration
        cal = results['calibration_metrics']
        auc_brier = ['AUC', 'Brier Score (inv)']
        auc_brier_values = [cm['auc'], 1 - cal['brier_score']]  # Invert Brier for better visualization
        
        bars2 = ax2.bar(auc_brier, auc_brier_values, color=['#9467bd', '#8c564b'])
        ax2.set_title('AUC & Calibration Quality')
        ax2.set_ylabel('Score')
        ax2.set_ylim(0, 1)
        
        for bar, value, orig_val in zip(bars2, auc_brier_values, [cm['auc'], cal['brier_score']]):
            if 'Brier' in auc_brier[list(bars2).index(bar)]:
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                        f'{orig_val:.3f}', ha='center', va='bottom')
            else:
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                        f'{value:.3f}', ha='center', va='bottom')
        
        # Error breakdown
        err = results['error_analysis']
        error_types = ['Correct', 'False Pos', 'False Neg']
        error_values = [err['correct_rate'], err['false_positive_rate'], err['false_negative_rate']]
        
        bars3 = ax3.bar(error_types, error_values, color=['#2ca02c', '#d62728', '#ff7f0e'])
        ax3.set_title('Prediction Breakdown')
        ax3.set_ylabel('Rate')
        ax3.set_ylim(0, 1)
        
        for bar, value in zip(bars3, error_values):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{value:.3f}', ha='center', va='bottom')
        
        # Confidence stats - create a simple bar chart instead of boxplot
        conf = results['confidence_analysis']['confidence_stats']
        conf_labels = ['Min', 'Q25', 'Median', 'Q75', 'Max']
        conf_values = [conf['min'], conf['q25'], conf['median'], conf['q75'], conf['max']]
        
        bars4 = ax4.bar(conf_labels, conf_values, color='lightblue', edgecolor='black')
        ax4.set_title('Confidence Distribution Stats')
        ax4.set_ylabel('Confidence')
        ax4.set_ylim(0, 1)
        
        # Add value labels
        for bar, value in zip(bars4, conf_values):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{value:.3f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'metrics_overview.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_roc_curve(self, results: Dict[str, Any], save_dir: str, model_name: str):
        """Plot ROC curve."""
        
        cm = results['classification_metrics']
        if not cm['fpr'] or not cm['tpr']:
            logger.warning("No ROC data available for plotting")
            return
        
        plt.figure(figsize=(8, 6))
        plt.plot(cm['fpr'], cm['tpr'], color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {cm["auc"]:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{model_name} - ROC Curve')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        plt.savefig(os.path.join(save_dir, 'roc_curve.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_precision_recall_curve(self, results: Dict[str, Any], save_dir: str, model_name: str):
        """Plot Precision-Recall curve."""
        
        cm = results['classification_metrics']
        if not cm['pr_precision'] or not cm['pr_recall']:
            logger.warning("No PR curve data available for plotting")
            return
        
        plt.figure(figsize=(8, 6))
        plt.plot(cm['pr_recall'], cm['pr_precision'], color='blue', lw=2,
                label=f'PR curve (AP = {cm["average_precision"]:.3f})')
        
        # Baseline (random classifier)
        baseline = results['n_positive'] / results['n_samples']
        plt.axhline(y=baseline, color='red', linestyle='--', 
                   label=f'Random (AP = {baseline:.3f})')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'{model_name} - Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        
        plt.savefig(os.path.join(save_dir, 'precision_recall_curve.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_calibration_curve(self, results: Dict[str, Any], save_dir: str, model_name: str):
        """Plot calibration curve."""
        
        cal = results['calibration_metrics']
        if not cal['calibration_curve']['fraction_of_positives']:
            logger.warning("No calibration data available for plotting")
            return
        
        plt.figure(figsize=(8, 6))
        
        fop = cal['calibration_curve']['fraction_of_positives']
        mpv = cal['calibration_curve']['mean_predicted_value']
        
        plt.plot(mpv, fop, "s-", color='blue', label=f'{model_name}')
        plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
        
        plt.xlabel('Mean Predicted Probability')
        plt.ylabel('Fraction of Positives')
        plt.title(f'{model_name} - Calibration Plot\n'
                 f'Brier Score: {cal["brier_score"]:.3f}, '
                 f'ECE: {cal["expected_calibration_error"]:.3f}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.savefig(os.path.join(save_dir, 'calibration_curve.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_confusion_matrix(self, results: Dict[str, Any], save_dir: str, model_name: str):
        """Plot confusion matrix."""
        
        cm_data = results['classification_metrics']['confusion_matrix']
        if not cm_data:
            logger.warning("No confusion matrix data available")
            return
        
        plt.figure(figsize=(8, 6))
        
        if sns is not None:
            sns.heatmap(cm_data, annot=True, fmt='d', cmap='Blues',
                       xticklabels=['Incorrect', 'Correct'],
                       yticklabels=['Incorrect', 'Correct'])
        else:
            plt.imshow(cm_data, interpolation='nearest', cmap='Blues')
            plt.colorbar()
            
            # Add text annotations
            for i in range(len(cm_data)):
                for j in range(len(cm_data[0])):
                    plt.text(j, i, str(cm_data[i][j]), ha='center', va='center')
            
            plt.xticks([0, 1], ['Incorrect', 'Correct'])
            plt.yticks([0, 1], ['Incorrect', 'Correct'])
        
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'{model_name} - Confusion Matrix')
        
        plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_confidence_distribution(self, y_pred_proba: np.ndarray, y_true: np.ndarray, 
                                    save_dir: str, model_name: str):
        """Plot confidence distribution."""
        
        plt.figure(figsize=(12, 5))
        
        # Overall distribution
        plt.subplot(1, 2, 1)
        plt.hist(y_pred_proba, bins=30, alpha=0.7, color='blue', edgecolor='black')
        plt.xlabel('Predicted Probability')
        plt.ylabel('Frequency')
        plt.title('Overall Confidence Distribution')
        plt.grid(True, alpha=0.3)
        
        # Distribution by class
        plt.subplot(1, 2, 2)
        correct_probs = y_pred_proba[y_true == 1]
        incorrect_probs = y_pred_proba[y_true == 0]
        
        plt.hist(incorrect_probs, bins=20, alpha=0.7, label='Incorrect (0)', color='red')
        plt.hist(correct_probs, bins=20, alpha=0.7, label='Correct (1)', color='green')
        plt.xlabel('Predicted Probability')
        plt.ylabel('Frequency')
        plt.title('Confidence by True Class')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.suptitle(f'{model_name} - Confidence Analysis')
        plt.tight_layout()
        
        plt.savefig(os.path.join(save_dir, 'confidence_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_confidence_vs_accuracy(self, results: Dict[str, Any], save_dir: str, model_name: str):
        """Plot confidence vs accuracy analysis."""
        
        conf_acc = results['confidence_analysis']['confidence_vs_accuracy']
        
        plt.figure(figsize=(10, 6))
        
        bin_centers = []
        boundaries = conf_acc['bin_boundaries']
        for i in range(len(boundaries) - 1):
            bin_centers.append((boundaries[i] + boundaries[i + 1]) / 2)
        
        # Filter out bins with zero counts to avoid dimension mismatch
        valid_bins = []
        valid_accuracies = []
        valid_counts = []
        
        for i, (center, accuracy, count) in enumerate(zip(bin_centers, conf_acc['bin_accuracies'], conf_acc['bin_counts'])):
            if count > 0:  # Only include bins with samples
                valid_bins.append(center)
                valid_accuracies.append(accuracy)
                valid_counts.append(count)
        
        if not valid_bins:
            logger.warning("No valid bins for confidence vs accuracy plot")
            return
        
        # Bar plot of accuracy by confidence bin
        plt.subplot(1, 2, 1)
        bars = plt.bar(valid_bins, valid_accuracies, 
                      width=0.08, alpha=0.7, color='skyblue', edgecolor='black')
        plt.plot([0, 1], [0, 1], 'r--', label='Perfect Calibration')
        plt.xlabel('Confidence Bin Center')
        plt.ylabel('Accuracy in Bin')
        plt.title('Accuracy vs Confidence')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Sample count per bin
        plt.subplot(1, 2, 2)
        plt.bar(valid_bins, valid_counts, 
               width=0.08, alpha=0.7, color='lightcoral', edgecolor='black')
        plt.xlabel('Confidence Bin Center')
        plt.ylabel('Sample Count')
        plt.title('Sample Distribution by Confidence')
        plt.grid(True, alpha=0.3)
        
        plt.suptitle(f'{model_name} - Confidence vs Accuracy Analysis')
        plt.tight_layout()
        
        plt.savefig(os.path.join(save_dir, 'confidence_vs_accuracy.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_error_analysis(self, results: Dict[str, Any], y_pred_proba: np.ndarray, 
                           y_true: np.ndarray, save_dir: str, model_name: str):
        """Plot error analysis."""
        
        y_pred = (y_pred_proba >= 0.5).astype(int)
        
        # Identify prediction types
        correct = (y_true == y_pred)
        false_pos = (y_true == 0) & (y_pred == 1)
        false_neg = (y_true == 1) & (y_pred == 0)
        
        plt.figure(figsize=(12, 8))
        
        # Error type distribution
        plt.subplot(2, 2, 1)
        error_counts = [np.sum(correct), np.sum(false_pos), np.sum(false_neg)]
        error_labels = ['Correct', 'False Positive', 'False Negative']
        colors = ['green', 'red', 'orange']
        
        plt.pie(error_counts, labels=error_labels, colors=colors, autopct='%1.1f%%')
        plt.title('Prediction Type Distribution')
        
        # Confidence by prediction type
        plt.subplot(2, 2, 2)
        conf_data = []
        conf_labels = []
        
        if np.sum(correct) > 0:
            conf_data.append(y_pred_proba[correct])
            conf_labels.append('Correct')
        if np.sum(false_pos) > 0:
            conf_data.append(y_pred_proba[false_pos])
            conf_labels.append('False Pos')
        if np.sum(false_neg) > 0:
            conf_data.append(y_pred_proba[false_neg])
            conf_labels.append('False Neg')
        
        if conf_data:
            plt.boxplot(conf_data, labels=conf_labels)
            plt.ylabel('Confidence')
            plt.title('Confidence by Prediction Type')
            plt.grid(True, alpha=0.3)
        
        # Error confidence histograms
        plt.subplot(2, 2, 3)
        if np.sum(false_pos) > 0:
            plt.hist(y_pred_proba[false_pos], bins=15, alpha=0.7, 
                    color='red', label='False Positives')
        if np.sum(false_neg) > 0:
            plt.hist(y_pred_proba[false_neg], bins=15, alpha=0.7, 
                    color='orange', label='False Negatives')
        plt.xlabel('Confidence')
        plt.ylabel('Count')
        plt.title('Error Confidence Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Summary statistics
        plt.subplot(2, 2, 4)
        err = results['error_analysis']
        
        stats_text = f"""Error Analysis Summary:
        
Correct Rate: {err['correct_rate']:.3f}
False Positive Rate: {err['false_positive_rate']:.3f}
False Negative Rate: {err['false_negative_rate']:.3f}

Confidence Statistics:"""
        
        if 'correct_confidence_mean' in err:
            stats_text += f"\nCorrect Predictions: {err['correct_confidence_mean']:.3f} ± {err['correct_confidence_std']:.3f}"
        if 'false_positive_confidence_mean' in err:
            stats_text += f"\nFalse Positives: {err['false_positive_confidence_mean']:.3f} ± {err['false_positive_confidence_std']:.3f}"
        if 'false_negative_confidence_mean' in err:
            stats_text += f"\nFalse Negatives: {err['false_negative_confidence_mean']:.3f} ± {err['false_negative_confidence_std']:.3f}"
        
        plt.text(0.1, 0.5, stats_text, transform=plt.gca().transAxes, 
                fontsize=10, verticalalignment='center', fontfamily='monospace')
        plt.axis('off')
        
        plt.suptitle(f'{model_name} - Error Analysis')
        plt.tight_layout()
        
        plt.savefig(os.path.join(save_dir, 'error_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_per_topic_analysis(self, results: Dict[str, Any], save_dir: str, model_name: str):
        """Plot per-topic performance analysis."""
        
        per_topic = results['per_topic_metrics']
        if 'summary' in per_topic:
            per_topic = {k: v for k, v in per_topic.items() if k != 'summary'}
        
        if not per_topic:
            logger.warning("No per-topic data available for plotting")
            return
        
        # Prepare data
        topics = list(per_topic.keys())
        accuracies = [per_topic[topic]['accuracy'] for topic in topics]
        f1_scores = [per_topic[topic]['f1'] for topic in topics]
        sample_counts = [per_topic[topic]['n_samples'] for topic in topics]
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Accuracy by topic
        bars1 = ax1.bar(range(len(topics)), accuracies, color='skyblue', edgecolor='black')
        ax1.set_xlabel('Topics')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Accuracy by Topic')
        ax1.set_xticks(range(len(topics)))
        ax1.set_xticklabels(topics, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, acc in zip(bars1, accuracies):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{acc:.3f}', ha='center', va='bottom', fontsize=8)
        
        # F1 scores by topic
        bars2 = ax2.bar(range(len(topics)), f1_scores, color='lightcoral', edgecolor='black')
        ax2.set_xlabel('Topics')
        ax2.set_ylabel('F1 Score')
        ax2.set_title('F1 Score by Topic')
        ax2.set_xticks(range(len(topics)))
        ax2.set_xticklabels(topics, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        
        for bar, f1 in zip(bars2, f1_scores):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{f1:.3f}', ha='center', va='bottom', fontsize=8)
        
        # Sample distribution
        bars3 = ax3.bar(range(len(topics)), sample_counts, color='lightgreen', edgecolor='black')
        ax3.set_xlabel('Topics')
        ax3.set_ylabel('Sample Count')
        ax3.set_title('Sample Count by Topic')
        ax3.set_xticks(range(len(topics)))
        ax3.set_xticklabels(topics, rotation=45, ha='right')
        ax3.grid(True, alpha=0.3)
        
        for bar, count in zip(bars3, sample_counts):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    f'{count}', ha='center', va='bottom', fontsize=8)
        
        # Performance correlation
        ax4.scatter(sample_counts, accuracies, alpha=0.7, s=100, color='purple')
        ax4.set_xlabel('Sample Count')
        ax4.set_ylabel('Accuracy')
        ax4.set_title('Accuracy vs Sample Count')
        ax4.grid(True, alpha=0.3)
        
        # Add topic labels to scatter plot
        for i, topic in enumerate(topics):
            ax4.annotate(topic, (sample_counts[i], accuracies[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        plt.suptitle(f'{model_name} - Per-Topic Performance Analysis')
        plt.tight_layout()
        
        plt.savefig(os.path.join(save_dir, 'per_topic_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_model_comparison(self, model_results: List[Dict[str, Any]], save_dir: str = None):
        """
        Plot comparison between multiple models.
        
        Args:
            model_results: List of evaluation result dictionaries
            save_dir: Directory to save comparison plots
        """
        if not PLOTTING_AVAILABLE:
            logger.warning("Matplotlib/seaborn not available. Skipping comparison plots.")
            return
        
        if len(model_results) < 2:
            logger.warning("Need at least 2 models for comparison")
            return
        
        if save_dir is None:
            save_dir = self.viz_dir
        
        # Prepare data
        model_names = [r['model_name'] for r in model_results]
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
        
        metric_data = {}
        for metric in metrics:
            metric_data[metric] = []
            for result in model_results:
                if metric in result['classification_metrics']:
                    metric_data[metric].append(result['classification_metrics'][metric])
                else:
                    metric_data[metric].append(0.0)
        
        # Create comparison plot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Bar chart comparison
        x = np.arange(len(model_names))
        width = 0.15
        
        for i, metric in enumerate(metrics):
            ax1.bar(x + i * width, metric_data[metric], width, 
                   label=metric.capitalize(), alpha=0.8)
        
        ax1.set_xlabel('Models')
        ax1.set_ylabel('Score')
        ax1.set_title('Model Performance Comparison')
        ax1.set_xticks(x + width * 2)
        ax1.set_xticklabels(model_names, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Radar chart (if we have enough metrics)
        if len(metrics) >= 3:
            angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
            angles += angles[:1]  # Complete the circle
            
            ax2 = plt.subplot(2, 2, 2, projection='polar')
            
            for i, model_name in enumerate(model_names):
                values = [metric_data[metric][i] for metric in metrics]
                values += values[:1]  # Complete the circle
                
                ax2.plot(angles, values, 'o-', linewidth=2, label=model_name)
                ax2.fill(angles, values, alpha=0.25)
            
            ax2.set_xticks(angles[:-1])
            ax2.set_xticklabels([m.capitalize() for m in metrics])
            ax2.set_ylim(0, 1)
            ax2.set_title('Performance Radar Chart')
            ax2.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        # Calibration comparison
        brier_scores = []
        for result in model_results:
            if 'calibration_metrics' in result:
                brier_scores.append(result['calibration_metrics']['brier_score'])
            else:
                brier_scores.append(0.0)
        
        bars = ax3.bar(model_names, brier_scores, color='orange', alpha=0.7)
        ax3.set_xlabel('Models')
        ax3.set_ylabel('Brier Score (lower is better)')
        ax3.set_title('Calibration Quality Comparison')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3)
        
        for bar, score in zip(bars, brier_scores):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
                    f'{score:.3f}', ha='center', va='bottom')
        
        # Summary statistics table
        ax4.axis('tight')
        ax4.axis('off')
        
        table_data = []
        for i, model_name in enumerate(model_names):
            row = [model_name]
            for metric in metrics:
                row.append(f"{metric_data[metric][i]:.3f}")
            table_data.append(row)
        
        table = ax4.table(cellText=table_data,
                         colLabels=['Model'] + [m.capitalize() for m in metrics],
                         cellLoc='center',
                         loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        ax4.set_title('Performance Summary Table')
        
        plt.suptitle('Model Comparison Analysis')
        plt.tight_layout()
        
        plt.savefig(os.path.join(save_dir, 'model_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Model comparison visualization saved to {save_dir}")
