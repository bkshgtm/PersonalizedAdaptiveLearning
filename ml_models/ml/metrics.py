import numpy as np
import logging
from typing import Dict, List, Tuple, Any, Optional
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

logger = logging.getLogger(__name__)


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """
    Calculate performance metrics for binary classification.
    
    Args:
        y_true: Array of true labels (0 or 1)
        y_pred: Array of predicted labels (0 or 1)
        y_prob: Array of predicted probabilities (0.0 to 1.0)
        
    Returns:
        Dictionary of metrics including accuracy, AUC, precision, recall, and F1
    """
    metrics = {}
    
    # Calculate accuracy
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    
    # Calculate AUC
    try:
        metrics['auc'] = roc_auc_score(y_true, y_prob)
    except Exception as e:
        logger.warning(f"Failed to calculate AUC: {str(e)}")
        metrics['auc'] = 0.0
    
    # Calculate precision, recall, and F1
    metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
    metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)
    
    return metrics


def calculate_rmse(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Calculate Root Mean Squared Error.
    
    Args:
        y_true: Array of true labels (0 or 1)
        y_prob: Array of predicted probabilities (0.0 to 1.0)
        
    Returns:
        RMSE value
    """
    return np.sqrt(np.mean((y_true - y_prob) ** 2))


def calculate_model_performance(
    model_predictions: Dict[str, Dict[int, float]],
    true_performances: Dict[str, Dict[int, Dict[str, Any]]]
) -> Dict[str, Any]:
    """
    Calculate overall model performance based on predictions for students across all topics.
    
    Args:
        model_predictions: Dictionary mapping student IDs to another dictionary 
                          mapping topic IDs to predicted probabilities
        true_performances: Dictionary mapping student IDs to another dictionary
                          mapping topic IDs to dictionaries with 'accuracy' and other metrics
                          
    Returns:
        Dictionary with overall performance metrics
    """
    all_true = []
    all_pred_probs = []
    
    # Aggregate all predictions and true values
    for student_id, topic_preds in model_predictions.items():
        if student_id not in true_performances:
            continue
            
        true_perf = true_performances[student_id]
        
        for topic_id, pred_prob in topic_preds.items():
            if topic_id not in true_perf:
                continue
                
            # Get the true accuracy for this topic
            true_accuracy = true_perf[topic_id].get('accuracy', 0)
            
            # Add to lists
            all_true.append(true_accuracy)
            all_pred_probs.append(pred_prob)
    
    # Convert to numpy arrays
    all_true = np.array(all_true)
    all_pred_probs = np.array(all_pred_probs)
    
    # Convert true accuracies to binary (>= 0.5 is considered mastered)
    all_true_binary = (all_true >= 0.5).astype(int)
    all_pred_binary = (all_pred_probs >= 0.5).astype(int)
    
    # Calculate metrics
    binary_metrics = calculate_metrics(all_true_binary, all_pred_binary, all_pred_probs)
    rmse = calculate_rmse(all_true, all_pred_probs)
    
    # Combine metrics
    metrics = {
        'rmse': rmse,
        **binary_metrics
    }
    
    return metrics


def get_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, int]:
    """
    Calculate confusion matrix values.
    
    Args:
        y_true: Array of true labels (0 or 1)
        y_pred: Array of predicted labels (0 or 1)
        
    Returns:
        Dictionary with confusion matrix values (TP, FP, TN, FN)
    """
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    return {
        'true_positives': int(tp),
        'false_positives': int(fp),
        'true_negatives': int(tn),
        'false_negatives': int(fn)
    }


def calculate_topic_metrics(
    student_interactions: List[Dict],
    topic_dict: Dict[int, int]
) -> Dict[int, Dict[str, Any]]:
    """
    Calculate performance metrics for each topic based on student interactions.
    
    Args:
        student_interactions: List of dictionaries with student interaction data
        topic_dict: Dictionary mapping topic IDs to indices
        
    Returns:
        Dictionary mapping topic IDs to dictionaries of metrics
    """
    # Group interactions by topic
    topic_interactions = {}
    
    for interaction in student_interactions:
        topic_id = interaction['topic_id']
        
        if topic_id not in topic_interactions:
            topic_interactions[topic_id] = []
            
        topic_interactions[topic_id].append(interaction)
    
    # Calculate metrics for each topic
    topic_metrics = {}
    
    for topic_id, interactions in topic_interactions.items():
        # Extract correctness values
        correctness = [1 if interaction['correct'] else 0 for interaction in interactions]
        
        # Calculate basic metrics
        num_interactions = len(correctness)
        num_correct = sum(correctness)
        accuracy = num_correct / num_interactions if num_interactions > 0 else 0
        
        # Add to result
        topic_metrics[topic_id] = {
            'num_interactions': num_interactions,
            'num_correct': num_correct,
            'accuracy': accuracy
        }
    
    return topic_metrics


def calculate_student_topic_metrics(
    student_interactions: List[Dict]
) -> Dict[str, Dict[int, Dict[str, Any]]]:
    """
    Calculate performance metrics for each student and topic.
    
    Args:
        student_interactions: List of dictionaries with student interaction data
        
    Returns:
        Dictionary mapping student IDs to dictionaries 
        mapping topic IDs to dictionaries of metrics
    """
    # Group interactions by student and topic
    student_topic_interactions = {}
    
    for interaction in student_interactions:
        student_id = interaction['student_id']
        topic_id = interaction['topic_id']
        
        if student_id not in student_topic_interactions:
            student_topic_interactions[student_id] = {}
            
        if topic_id not in student_topic_interactions[student_id]:
            student_topic_interactions[student_id][topic_id] = []
            
        student_topic_interactions[student_id][topic_id].append(interaction)
    
    # Calculate metrics for each student and topic
    student_topic_metrics = {}
    
    for student_id, topic_interactions in student_topic_interactions.items():
        student_topic_metrics[student_id] = {}
        
        for topic_id, interactions in topic_interactions.items():
            # Extract correctness values
            correctness = [1 if interaction['correct'] else 0 for interaction in interactions]
            
            # Calculate basic metrics
            num_interactions = len(correctness)
            num_correct = sum(correctness)
            accuracy = num_correct / num_interactions if num_interactions > 0 else 0
            
            # Calculate trend
            trend = 'stagnant'
            if num_interactions >= 3:
                # Compare accuracy of first half vs second half
                mid_idx = num_interactions // 2
                first_half = correctness[:mid_idx]
                second_half = correctness[mid_idx:]
                
                first_acc = sum(first_half) / len(first_half) if len(first_half) > 0 else 0
                second_acc = sum(second_half) / len(second_half) if len(second_half) > 0 else 0
                
                if second_acc > first_acc + 0.1:
                    trend = 'improving'
                elif second_acc < first_acc - 0.1:
                    trend = 'declining'
            
            # Add to result
            student_topic_metrics[student_id][topic_id] = {
                'num_interactions': num_interactions,
                'num_correct': num_correct,
                'accuracy': accuracy,
                'trend': trend,
                'interactions': interactions
            }
    
    return student_topic_metrics