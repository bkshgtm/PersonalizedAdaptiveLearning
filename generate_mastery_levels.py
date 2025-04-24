import os
import django
import logging
import torch
from django.utils import timezone
from django.db import transaction

# Set up Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'pal_project.settings')
django.setup()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from ml_models.models import KnowledgeTracingModel, TrainingJob, PredictionBatch, TopicMastery
from ml_models.ml.data_preparation import DataPreparation
from ml_models.ml.dkt import DKTModel, DKTTrainer
from ml_models.ml.sakt import SAKTModel, SAKTTrainer
from ml_models.ml.metrics import calculate_student_topic_metrics
from core.models import Course, Student, Topic
from django.contrib.auth.models import User


def train_model_for_course(course_id, model_type='dkt'):
    """
    Train a knowledge tracing model for a course.
    
    Args:
        course_id: ID of the course
        model_type: Type of model to train ('dkt' or 'sakt')
        
    Returns:
        The trained model record if successful, None otherwise
    """
    logger.info(f"Training {model_type.upper()} model for course {course_id}")
    
    # Get the course
    try:
        course = Course.objects.get(course_id=course_id)
        logger.info(f"Found course: {course.title}")
    except Course.DoesNotExist:
        logger.error(f"Course {course_id} not found")
        return None
    
    # Get an admin user
    admin_user = User.objects.filter(is_staff=True).first()
    if not admin_user:
        logger.error("No admin user found")
        return None
    
    # Create model directory
    models_dir = os.path.join('models', 'knowledge_tracing', course_id)
    os.makedirs(models_dir, exist_ok=True)
    
    # Create model path
    model_path = os.path.join(models_dir, f'{model_type}_model.pt')
    
    # Set hyperparameters based on model type
    if model_type == 'dkt':
        hyperparameters = {
            'hidden_size': 100,
            'num_layers': 1,
            'dropout': 0.2,
            'batch_size': 32,
            'learning_rate': 0.001,
            'num_epochs': 10
        }
    else:  # sakt
        hyperparameters = {
            'd_model': 64,
            'n_heads': 8,
            'dropout': 0.2,
            'batch_size': 32,
            'learning_rate': 0.001,
            'num_epochs': 10
        }
    
    # Check if model already exists
    existing_model = KnowledgeTracingModel.objects.filter(
        model_type=model_type,
        course=course,
        is_default=True
    ).first()
    
    if existing_model:
        logger.info(f"Using existing model: {existing_model.name}")
        model_record = existing_model
    else:
        # Create the model record
        model_record = KnowledgeTracingModel.objects.create(
            name=f"Default {model_type.upper()} for {course.title}",
            model_type=model_type,
            description=f"Default {model_type.upper()} model",
            created_by=admin_user,
            model_path=model_path,
            hyperparameters=hyperparameters,
            status='created',
            is_default=True,
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
        return None
    
    # Update job with data stats
    job.total_interactions = len(interactions)
    
    # Get unique students
    student_ids = set(interaction['student_id'] for interaction in interactions)
    job.total_students = len(student_ids)
    job.save(update_fields=['total_interactions', 'total_students'])
    
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
        return None
    
    # Update job status
    job.status = 'processing'
    job.started_at = timezone.now()
    job.save(update_fields=['status', 'started_at'])
    
    try:
        # Train the model
        num_epochs = hyperparameters.get('num_epochs', 10)
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
        
        # Update model status
        model_record.status = 'active'
        model_record.save(update_fields=['status'])
        
        logger.info(f"Training completed successfully")
        logger.info(f"Final test accuracy: {history['test_accuracy'][-1]:.4f}")
        logger.info(f"Final test AUC: {history['test_auc'][-1]:.4f}")
        
        return model_record
        
    except Exception as e:
        logger.exception(f"Error in training: {str(e)}")
        
        # Update job status
        job.status = 'failed'
        job.error_message = str(e)
        job.completed_at = timezone.now()
        job.save(update_fields=['status', 'error_message', 'completed_at'])
        
        return None


def generate_predictions(model_record):
    """
    Generate mastery predictions for all students using a trained model.
    
    Args:
        model_record: KnowledgeTracingModel instance
        
    Returns:
        The prediction batch if successful, None otherwise
    """
    logger.info(f"Generating predictions using model: {model_record.name}")
    
    # Create prediction batch
    batch = PredictionBatch.objects.create(
        model=model_record,
        status='pending'
    )
    
    # Update batch status
    batch.status = 'processing'
    batch.started_at = timezone.now()
    batch.save(update_fields=['status', 'started_at'])
    
    try:
        # Get the course
        course = model_record.course
        
        # Load the model
        if model_record.model_type == 'dkt':
            model, topic_ids, _ = DKTModel.load(model_record.model_path)
        elif model_record.model_type == 'sakt':
            model, topic_ids, _ = SAKTModel.load(model_record.model_path)
        else:
            raise ValueError(f"Unsupported model type: {model_record.model_type}")
        
        # Prepare data
        data_prep = DataPreparation(course.course_id)
        
        # Get topic mapping
        topic_dict, _ = data_prep.get_topic_mapping()
        
        # Get all students for the course
        students = Student.objects.filter(courses=course)
        
        # Update batch with total students
        batch.total_students = students.count()
        batch.save(update_fields=['total_students'])
        
        # Get all topics for the course
        topics = Topic.objects.filter(course=course)
        
        # Get all student interactions
        all_interactions = data_prep.get_interactions()
        
        # Calculate student-topic metrics for trend analysis
        student_topic_metrics = calculate_student_topic_metrics(all_interactions)
        
        # Process each student
        for i, student in enumerate(students):
            try:
                # Prepare input data for prediction
                input_data = data_prep.prepare_for_prediction(student.student_id, topic_dict)
                
                if input_data is None or input_data['seq_len'] == 0:
                    # No interaction data for this student, skip
                    logger.warning(f"No interaction data for student {student.student_id}")
                    continue
                
                # Get input tensors
                input_ids = input_data['input_ids']
                input_labels = input_data.get('labels')  # Use get() to handle case where labels might not exist
                
                # Generate predictions for all topics
                topic_predictions = model.predict(input_ids, input_labels, topic_ids)
                
                # Create mastery records
                mastery_records = []
                
                for topic in topics:
                    # Get the prediction for this topic
                    mastery_score = topic_predictions.get(topic.id, 0.5)  # Default to 0.5 if not predicted
                    
                    # Get confidence (fixed for now)
                    confidence = 0.8
                    
                    # Determine trend
                    trend = 'stagnant'
                    if student.student_id in student_topic_metrics and topic.id in student_topic_metrics[student.student_id]:
                        trend = student_topic_metrics[student.student_id][topic.id]['trend']
                    
                    # Create trend data (historical scores)
                    trend_data = []
                    
                    # Create mastery record
                    mastery = TopicMastery(
                        student=student,
                        topic=topic,
                        prediction_batch=batch,
                        mastery_score=mastery_score,
                        confidence=confidence,
                        trend=trend,
                        trend_data=trend_data
                    )
                    
                    mastery_records.append(mastery)
                
                # Bulk create mastery records
                with transaction.atomic():
                    TopicMastery.objects.bulk_create(mastery_records)
                
                # Update batch progress
                batch.processed_students += 1
                if i % 10 == 0:  # Update every 10 students to reduce database writes
                    batch.save(update_fields=['processed_students'])
                    logger.info(f"Processed {batch.processed_students}/{batch.total_students} students")
            
            except Exception as student_error:
                logger.error(f"Error processing student {student.student_id}: {str(student_error)}")
        
        # Ensure final count is saved
        batch.processed_students = i + 1
        
        # Update batch status
        batch.status = 'completed'
        batch.completed_at = timezone.now()
        batch.save(update_fields=['processed_students', 'status', 'completed_at'])
        
        logger.info(f"Prediction batch completed successfully")
        logger.info(f"Processed {batch.processed_students} students")
        
        return batch
        
    except Exception as e:
        logger.exception(f"Error in prediction batch: {str(e)}")
        
        # Update batch status
        batch.status = 'failed'
        batch.error_message = str(e)
        batch.completed_at = timezone.now()
        batch.save(update_fields=['status', 'error_message', 'completed_at'])
        
        return None


def generate_mastery_levels(course_id, model_type='dkt'):
    """
    Generate mastery levels for all students in a course.
    
    Args:
        course_id: ID of the course
        model_type: Type of model to use ('dkt' or 'sakt')
    """
    # Train the model
    model_record = train_model_for_course(course_id, model_type)
    
    if model_record is None:
        logger.error(f"Failed to train model for course {course_id}")
        return
    
    # Generate predictions
    batch = generate_predictions(model_record)
    
    if batch is None:
        logger.error(f"Failed to generate predictions for course {course_id}")
        return
    
    logger.info(f"Successfully generated mastery levels for course {course_id}")
    logger.info(f"Prediction batch ID: {batch.id}")


if __name__ == '__main__':
    # Generate mastery levels for CS206 course using DKT model
    generate_mastery_levels('CS206', 'dkt')
    
    # Generate mastery levels for CS206 course using SAKT model
    generate_mastery_levels('CS206', 'sakt')
