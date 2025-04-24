from celery import shared_task
import logging
import time
import torch
import numpy as np
from django.utils import timezone
from django.db import transaction

from .models import KnowledgeTracingModel, TrainingJob, PredictionBatch, TopicMastery
from .ml.data_preparation import DataPreparation
from .ml.dkt import DKTModel, DKTTrainer
from .ml.sakt import SAKTModel, SAKTTrainer
from .ml.metrics import calculate_student_topic_metrics
from core.models import Topic, Student

logger = logging.getLogger(__name__)


@shared_task
def train_knowledge_tracing_model(training_job_id: int):
    """
    Celery task to train a knowledge tracing model.
    
    Args:
        training_job_id: ID of the TrainingJob to process
    """
    try:
        # Get the training job
        job = TrainingJob.objects.get(pk=training_job_id)
        
        # Update job status
        job.status = 'processing'
        job.started_at = timezone.now()
        job.save(update_fields=['status', 'started_at'])
        
        # Get the model
        model_record = job.model
        
        # Get hyperparameters
        hyperparams = job.hyperparameters
        
        # Prepare data
        data_prep = DataPreparation(model_record.course.course_id)
        
        # Get topic mapping
        topic_dict, topic_ids = data_prep.get_topic_mapping()
        
        # Get all interactions
        interactions = data_prep.get_interactions()
        
        # Log the number of interactions
        logger.info(f"Total interactions: {len(interactions)}")
        
        # Update job with data stats
        job.total_interactions = len(interactions)
        
        # Get unique students
        student_ids = set(interaction['student_id'] for interaction in interactions)
        job.total_students = len(student_ids)
        job.save(update_fields=['total_interactions', 'total_students'])
        
        # Check if we have enough data
        if len(interactions) < 10:
            raise ValueError(f"Not enough interactions to train model: {len(interactions)}")
        
        # Split data into training and testing
        train_interactions, test_interactions = data_prep.split_data(
            interactions, 
            test_ratio=1.0 - job.split_ratio
        )
        
        # Log the split
        logger.info(f"Training interactions: {len(train_interactions)}")
        logger.info(f"Testing interactions: {len(test_interactions)}")
        
        # Create data loaders
        batch_size = hyperparams.get('batch_size', 32)
        max_seq_len = hyperparams.get('max_seq_len', 200)
        
        train_loader, test_loader = data_prep.create_dataloaders(
            train_interactions, 
            test_interactions, 
            topic_dict,
            batch_size=batch_size,
            max_seq_len=max_seq_len
        )
        
        # Choose device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Initialize model
        num_topics = len(topic_dict)
        
        if model_record.model_type == 'dkt':
            # DKT model
            hidden_size = hyperparams.get('hidden_size', 100)
            num_layers = hyperparams.get('num_layers', 1)
            dropout = hyperparams.get('dropout', 0.2)
            
            model = DKTModel(
                num_topics=num_topics,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout
            )
            
            # Create trainer
            learning_rate = hyperparams.get('learning_rate', 0.001)
            trainer = DKTTrainer(model, topic_ids, device, learning_rate)
            
        elif model_record.model_type == 'sakt':
            # SAKT model
            d_model = hyperparams.get('d_model', 64)
            n_heads = hyperparams.get('n_heads', 8)
            dropout = hyperparams.get('dropout', 0.2)
            
            model = SAKTModel(
                num_topics=num_topics,
                d_model=d_model,
                n_heads=n_heads,
                dropout=dropout
            )
            
            # Create trainer
            learning_rate = hyperparams.get('learning_rate', 0.001)
            trainer = SAKTTrainer(model, topic_ids, device, learning_rate)
            
        else:
            raise ValueError(f"Unsupported model type: {model_record.model_type}")
        
        # Train the model
        num_epochs = hyperparams.get('num_epochs', 10)
        history = trainer.train(train_loader, test_loader, num_epochs=num_epochs)
        
        # Save the model
        model.save(model_record.model_path, topic_ids, hyperparams)
        
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
        
        return True
        
    except Exception as e:
        logger.exception(f"Error in training job {training_job_id}: {str(e)}")
        
        # Update job status
        try:
            job = TrainingJob.objects.get(pk=training_job_id)
            job.status = 'failed'
            job.error_message = str(e)
            job.completed_at = timezone.now()
            job.save(update_fields=['status', 'error_message', 'completed_at'])
        except Exception as update_error:
            logger.error(f"Failed to update TrainingJob status: {str(update_error)}")
        
        # Re-raise the exception to mark the task as failed
        raise


@shared_task
def generate_mastery_predictions(prediction_batch_id: int):
    """
    Celery task to generate mastery predictions for all students in a course.
    
    Args:
        prediction_batch_id: ID of the PredictionBatch to process
    """
    try:
        # Get the prediction batch
        batch = PredictionBatch.objects.get(pk=prediction_batch_id)
        
        # Update batch status
        batch.status = 'processing'
        batch.started_at = timezone.now()
        batch.save(update_fields=['status', 'started_at'])
        
        # Get the model
        model_record = batch.model
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
                    continue
                
                # Get input tensors
                input_ids = input_data['input_ids']
                input_labels = input_data['labels']
                
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
            
            except Exception as student_error:
                logger.error(f"Error processing student {student.student_id}: {str(student_error)}")
        
        # Ensure final count is saved
        batch.processed_students = i + 1
        
        # Update batch status
        batch.status = 'completed'
        batch.completed_at = timezone.now()
        batch.save(update_fields=['processed_students', 'status', 'completed_at'])
        
        return True
        
    except Exception as e:
        logger.exception(f"Error in prediction batch {prediction_batch_id}: {str(e)}")
        
        # Update batch status
        try:
            batch = PredictionBatch.objects.get(pk=prediction_batch_id)
            batch.status = 'failed'
            batch.error_message = str(e)
            batch.completed_at = timezone.now()
            batch.save(update_fields=['status', 'error_message', 'completed_at'])
        except Exception as update_error:
            logger.error(f"Failed to update PredictionBatch status: {str(update_error)}")
        
        # Re-raise the exception to mark the task as failed
        raise


@shared_task
def create_default_models(course_id: str):
    """
    Celery task to create default knowledge tracing models for a course.
    
    Args:
        course_id: ID of the course
    """
    from core.models import Course
    from django.contrib.auth.models import User
    import os
    
    try:
        # Get the course
        course = Course.objects.get(course_id=course_id)
        
        # Get the first admin user
        admin_user = User.objects.filter(is_staff=True).first()
        
        if not admin_user:
            raise ValueError("No admin user found")
        
        # Create directory for models
        models_dir = os.path.join('models', 'knowledge_tracing', course_id)
        os.makedirs(models_dir, exist_ok=True)
        
        # Create DKT model
        dkt_model = KnowledgeTracingModel.objects.create(
            name=f"Default DKT for {course.title}",
            model_type='dkt',
            description="Default Deep Knowledge Tracing model",
            created_by=admin_user,
            model_path=os.path.join(models_dir, 'dkt_model.pt'),
            hyperparameters={
                'hidden_size': 100,
                'num_layers': 1,
                'dropout': 0.2,
                'batch_size': 32,
                'learning_rate': 0.001,
                'num_epochs': 10
            },
            status='created',
            is_default=True,
            course=course
        )
        
        # Create SAKT model
        sakt_model = KnowledgeTracingModel.objects.create(
            name=f"Default SAKT for {course.title}",
            model_type='sakt',
            description="Default Self-Attentive Knowledge Tracing model",
            created_by=admin_user,
            model_path=os.path.join(models_dir, 'sakt_model.pt'),
            hyperparameters={
                'd_model': 64,
                'n_heads': 8,
                'dropout': 0.2,
                'batch_size': 32,
                'learning_rate': 0.001,
                'num_epochs': 10
            },
            status='created',
            is_default=False,
            course=course
        )
        
        # Create training jobs for both models
        dkt_job = TrainingJob.objects.create(
            model=dkt_model,
            status='pending',
            hyperparameters=dkt_model.hyperparameters,
            split_ratio=0.8
        )
        
        sakt_job = TrainingJob.objects.create(
            model=sakt_model,
            status='pending',
            hyperparameters=sakt_model.hyperparameters,
            split_ratio=0.8
        )
        
        # Start training jobs
        train_knowledge_tracing_model.delay(dkt_job.id)
        
        # Wait for the DKT model to complete before starting SAKT
        # This helps prevent memory issues if running on a small server
        #train_knowledge_tracing_model.delay(sakt_job.id)
        
        return True
        
    except Exception as e:
        logger.exception(f"Error creating default models for course {course_id}: {str(e)}")
        raise
