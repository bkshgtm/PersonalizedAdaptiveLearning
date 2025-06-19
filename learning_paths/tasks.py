from celery import shared_task
import logging
from django.utils import timezone

# from .models import PathGenerationJob  # Old model, not needed with new architecture
# from .services.path_generator import LearningPathGenerator
from .ml.adaptive_path_lstm import DjangoIntegratedPathGenerator

logger = logging.getLogger(__name__)


@shared_task
def generate_learning_path(job_id: int):
    """
    Celery task to generate a learning path.
    
    Args:
        job_id: ID of the PathGenerationJob to process
    """
    try:
        logger.info(f"Starting learning path generation task for job {job_id}")
        
        # For now, this task is not used with the new architecture
        # Use the generate_learning_paths management command instead
        logger.warning(f"generate_learning_path task called but not implemented with new models")
        return False
        
    except Exception as e:
        logger.exception(f"Error in learning path generation task for job {job_id}: {str(e)}")
        raise


@shared_task
def refresh_learning_paths(course_id: str):
    """
    Celery task to refresh learning paths for all students in a course.
    
    Args:
        course_id: ID of the course to refresh paths for
    """
    from core.models import Course, Student
    from ml_models.models import PredictionBatch
    from knowledge_graph.models import KnowledgeGraph
    
    try:
        logger.info(f"Starting learning path refresh task for course {course_id}")
        
        # Get the course
        course = Course.objects.get(course_id=course_id)
        
        # Get all students in the course
        students = Student.objects.filter(courses=course)
        
        if not students:
            logger.warning(f"No students found for course {course_id}")
            return True
        
        # Get the latest prediction batch
        prediction_batch = PredictionBatch.objects.filter(
            model__course=course,
            status='completed'
        ).order_by('-completed_at').first()
        
        if not prediction_batch:
            logger.warning(f"No completed prediction batch found for course {course_id}")
            # Continue without prediction batch
        
        # Get the active knowledge graph
        knowledge_graph = KnowledgeGraph.objects.filter(is_active=True).first()
        
        if not knowledge_graph:
            logger.warning("No active knowledge graph found")
            # Continue without knowledge graph
        
        # For now, this task is not implemented with the new architecture
        # Use the generate_learning_paths management command instead
        logger.warning(f"refresh_learning_paths task called but not implemented with new models")
        logger.info(f"Use: python manage.py generate_learning_paths --all-students for course {course_id}")
        return False
        
    except Exception as e:
        logger.exception(f"Error in learning path refresh task for course {course_id}: {str(e)}")
        # Re-raise the exception to mark the task as failed
        raise
