from celery import shared_task
import logging

from django.utils import timezone

from .models import ClassificationJob, ClassificationModel
from core.models import Question

logger = logging.getLogger(__name__)

def _get_classifier(model_id):
    """Lazy import to avoid circular imports during URL loading"""
    from topic_classification.services.classifier_factory import get_classifier
    return get_classifier(model_id)

@shared_task
def classify_questions(job_id):
    """
    Celery task to classify questions in a batch.
    
    Args:
        job_id: ID of the ClassificationJob to process
    """
    try:
        logger.info(f"Starting question classification task for job {job_id}")
        
        # Get the job
        job = ClassificationJob.objects.get(pk=job_id)
        
        # Get the appropriate classifier (lazy-loaded)
        classifier = _get_classifier(job.model_id)
        
        # Process the batch
        success = classifier.classify_batch(job_id)
        
        if success:
            logger.info(f"Question classification completed for job {job_id}")
        else:
            logger.error(f"Question classification failed for job {job_id}")
        
        return success
    
    except Exception as e:
        logger.exception(f"Error in question classification task for job {job_id}: {str(e)}")
        
        # Update the job status to failed
        try:
            job = ClassificationJob.objects.get(pk=job_id)
            job.status = 'failed'
            job.error_message = str(e)
            job.completed_at = timezone.now()
            job.save(update_fields=['status', 'error_message', 'completed_at'])
        except Exception as update_error:
            logger.error(f"Failed to update ClassificationJob status: {str(update_error)}")
        
        # Re-raise the exception to mark the task as failed
        raise


@shared_task
def classify_new_questions():
    """
    Celery task to find and classify new questions without topics.
    """
    try:
        logger.info("Starting classification of new questions")
        
        # Find questions without topics
        questions = Question.objects.filter(topic__isnull=True)
        question_count = questions.count()
        
        if question_count == 0:
            logger.info("No new questions to classify")
            return True
        
        logger.info(f"Found {question_count} questions to classify")
        
        # Get the default classification model
        try:
            model = ClassificationModel.objects.get(is_default=True, status='active')
        except ClassificationModel.DoesNotExist:
            logger.error("No default active classification model found")
            return False
        
        # Create a new classification job
        job = ClassificationJob.objects.create(
            model=model,
            status='pending',
            total_questions=question_count
        )
        
        # Start the classification task
        classify_questions.delay(job.id)
        
        logger.info(f"Created classification job {job.id} for {question_count} questions")
        return True
    
    except Exception as e:
        logger.exception(f"Error in new questions classification task: {str(e)}")
        return False
