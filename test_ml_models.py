import os
import django
import logging
import torch
import numpy as np
from django.utils import timezone

# Set up Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'pal_project.settings')
django.setup()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from ml_models.ml.data_preparation import DataPreparation
from ml_models.ml.dkt import DKTModel, DKTTrainer
from ml_models.ml.sakt import SAKTModel, SAKTTrainer
from ml_models.models import KnowledgeTracingModel, TrainingJob
from core.models import Course, Student, Topic
from django.contrib.auth.models import User


def test_model_training(model_type='dkt', course_id='CS206'):
    """
    Test training a knowledge tracing model.
    
    Args:
        model_type: Type of model to train ('dkt' or 'sakt')
        course_id: ID of the course to use for training
    """
    logger.info(f"Testing {model_type.upper()} model training for course {course_id}")
    
    # Get the course
    try:
        course = Course.objects.get(course_id=course_id)
        logger.info(f"Found course: {course.title}")
    except Course.DoesNotExist:
        logger.error(f"Course {course_id} not found")
        return False
    
    # Get an admin user
    admin_user = User.objects.filter(is_staff=True).first()
    if not admin_user:
        logger.error("No admin user found")
        return False
    
    # Create model directory
    models_dir = os.path.join('models', 'knowledge_tracing', course_id)
    os.makedirs(models_dir, exist_ok=True)
    
    # Create model path
    model_path = os.path.join(models_dir, f'{model_type}_test_model.pt')
    
    # Set hyperparameters based on model type
    if model_type == 'dkt':
        hyperparameters = {
            'hidden_size': 100,
            'num_layers': 1,
            'dropout': 0.2,
            'batch_size': 32,
            'learning_rate': 0.001,
            'num_epochs': 20  # Increased for better training
        }
    else:  # sakt
        hyperparameters = {
            'd_model': 64,
            'n_heads': 8,
            'dropout': 0.2,
            'batch_size': 32,
            'learning_rate': 0.001,
            'num_epochs': 20  # Increased for better training
        }
    
    # Create the model record
    model_record = KnowledgeTracingModel.objects.create(
        name=f"Test {model_type.upper()} for {course.title}",
        model_type=model_type,
        description=f"Test {model_type.upper()} model",
        created_by=admin_user,
        model_path=model_path,
        hyperparameters=hyperparameters,
        status='created',
        is_default=False,
        course=course
    )
    
    # Create training job
    job = TrainingJob.objects.create(
        model=model_record,
        status='pending',
        hyperparameters=hyperparameters,
        split_ratio=0.8
    )
    
    # Prepare data
    data_prep = DataPreparation(course_id)
    
    # Get topic mapping
    topic_dict, topic_ids = data_prep.get_topic_mapping()
    logger.info(f"Found {len(topic_ids)} topics")
    
    # Get all interactions
    interactions = data_prep.get_interactions()
    logger.info(f"Found {len(interactions)} interactions")
    
    # Check if we have enough data
    if len(interactions) < 10:
        logger.error(f"Not enough interactions to train model: {len(interactions)}")
        return False
    
    # Split data into training and testing
    train_interactions, test_interactions = data_prep.split_data(
        interactions, 
        test_ratio=0.2
    )
    logger.info(f"Training interactions: {len(train_interactions)}")
    logger.info(f"Testing interactions: {len(test_interactions)}")
    
    # Create data loaders
    batch_size = hyperparameters.get('batch_size', 32)
    max_seq_len = hyperparameters.get('max_seq_len', 200)
    
    train_loader, test_loader = data_prep.create_dataloaders(
        train_interactions, 
        test_interactions, 
        topic_dict,
        batch_size=batch_size,
        max_seq_len=max_seq_len
    )
    
    # Choose device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    # Initialize model
    num_topics = len(topic_dict)
    
    if model_type == 'dkt':
        # DKT model
        hidden_size = hyperparameters.get('hidden_size', 100)
        num_layers = hyperparameters.get('num_layers', 1)
        dropout = hyperparameters.get('dropout', 0.2)
        
        model = DKTModel(
            num_topics=num_topics,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout
        )
        
        # Create trainer
        learning_rate = hyperparameters.get('learning_rate', 0.001)
        trainer = DKTTrainer(model, topic_ids, device, learning_rate)
        
    elif model_type == 'sakt':
        # SAKT model
        d_model = hyperparameters.get('d_model', 64)
        n_heads = hyperparameters.get('n_heads', 8)
        dropout = hyperparameters.get('dropout', 0.2)
        
        model = SAKTModel(
            num_topics=num_topics,
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout
        )
        
        # Create trainer
        learning_rate = hyperparameters.get('learning_rate', 0.001)
        trainer = SAKTTrainer(model, topic_ids, device, learning_rate)
        
    else:
        logger.error(f"Unsupported model type: {model_type}")
        return False
    
    # Update job status
    job.status = 'processing'
    job.started_at = timezone.now()
    job.save(update_fields=['status', 'started_at'])
    
    try:
        # Train the model
        num_epochs = hyperparameters.get('num_epochs', 5)
        history = trainer.train(train_loader, test_loader, num_epochs=num_epochs)
        
        # Save the model
        model.save(model_path, topic_ids, hyperparameters)
        logger.info(f"Model saved to {model_path}")
        
        # Update job with metrics
        job.metrics = {
            'train_loss': history['train_loss'],
            'test_loss': history['test_loss'],
            'test_accuracy': history['test_accuracy'],
            'test_auc': history['test_auc'],
            'final_test_loss': history['test_loss'][-1],
            'final_test_accuracy': history['test_accuracy'][-1],
            'final_test_auc': history['test_auc'][-1]
        }
        
        # Update job status
        job.status = 'completed'
        job.completed_at = timezone.now()
        job.save(update_fields=['metrics', 'status', 'completed_at'])
        
        # Update model status and set as default
        model_record.status = 'active'
        model_record.is_default = True
        model_record.save(update_fields=['status', 'is_default'])
        
        logger.info(f"Training completed successfully")
        logger.info(f"Final test accuracy: {history['test_accuracy'][-1]:.4f}")
        logger.info(f"Final test AUC: {history['test_auc'][-1]:.4f}")
        
        # Test prediction for a student
        students = Student.objects.filter(courses=course)
        if students.exists():
            student = students.first()
            logger.info(f"Testing prediction for student {student.student_id}")
            
            # Prepare input data for prediction
            input_data = data_prep.prepare_for_prediction(student.student_id, topic_dict)
            
            if input_data is not None and input_data['seq_len'] > 0:
                # Get input tensors
                input_ids = input_data['input_ids']
                input_labels = input_data.get('labels')  # Use get() to handle case where labels might not exist
                
                # Generate predictions for all topics
                topic_predictions = model.predict(input_ids, input_labels, topic_ids)
                
                # Print predictions for a few topics
                topics = Topic.objects.filter(id__in=list(topic_predictions.keys()))
                logger.info(f"All predicted topics: {[t.name for t in topics]}")
                # Show predictions for all topics
                for topic in sorted(topics, key=lambda x: x.name):
                    pred = topic_predictions.get(topic.id, 0.5)
                    logger.info(f"Topic: {topic.name}, Prediction: {pred:.4f}")
            else:
                logger.warning(f"No interaction data for student {student.student_id}")
        
        return True
        
    except Exception as e:
        logger.exception(f"Error in training: {str(e)}")
        
        # Update job status
        job.status = 'failed'
        job.error_message = str(e)
        job.completed_at = timezone.now()
        job.save(update_fields=['status', 'error_message', 'completed_at'])
        
        return False


if __name__ == '__main__':
    # Test DKT model
    test_model_training(model_type='dkt')
    
    # Test SAKT model
    test_model_training(model_type='sakt')
