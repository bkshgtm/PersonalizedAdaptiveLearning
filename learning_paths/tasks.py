from celery import shared_task
import logging
from django.utils import timezone

from .models import PathGenerationJob
from .services.path_generator import LearningPathGenerator

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
        
        generator = LearningPathGenerator(job_id)
        path = generator.generate_path()
        
        logger.info(f"Learning path generated successfully for job {job_id}")
        return True
        
    except Exception as e:
        logger.exception(f"Error in learning path generation task for job {job_id}: {str(e)}")
        
        # Update job status to failed
        try:
            job = PathGenerationJob.objects.get(pk=job_id)
            job.status = 'failed'
            job.error_message = str(e)
            job.completed_at = timezone.now()
            job.save(update_fields=['status', 'error_message', 'completed_at'])
        except Exception as update_error:
            logger.error(f"Failed to update PathGenerationJob status: {str(update_error)}")
        
        # Re-raise the exception to mark the task as failed
        raise


@shared_task
def refresh_learning_paths(course_id: str):
    """
    Celery task to refresh learning paths for all students in a course.
    
    Args:
        course_id: ID of the course to refresh paths for
    """
    from core.models import Course, Student
    from learning_paths.models import PathGenerator, PathGenerationJob
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
        
        # Get the active path generator
        generator = PathGenerator.objects.filter(is_active=True).first()
        
        if not generator:
            logger.warning("No active path generator found")
            # Create a default generator
            from django.contrib.auth.models import User
            admin = User.objects.filter(is_staff=True).first()
            
            if not admin:
                logger.error("No admin user found to create path generator")
                return False
                
            generator = PathGenerator.objects.create(
                name="Default Path Generator",
                description="Automatically created path generator",
                created_by=admin,
                is_active=True,
                config={
                    'include_strong_topics': False,
                    'max_resources_per_topic': 3
                }
            )
        
        # Create path generation jobs for each student
        jobs_created = 0
        
        for student in students:
            # Create a job for this student
            job = PathGenerationJob.objects.create(
                generator=generator,
                student=student,
                course=course,
                status='pending',
                prediction_batch=prediction_batch,
                knowledge_graph=knowledge_graph
            )
            
            # Start the generation task
            generate_learning_path.delay(job.id)
            
            jobs_created += 1
            
            # Log progress every 10 students
            if jobs_created % 10 == 0:
                logger.info(f"Created {jobs_created} path generation jobs for course {course_id}")
        
        logger.info(f"Learning path refresh completed for course {course_id}, {jobs_created} jobs created")
        return True
        
    except Exception as e:
        logger.exception(f"Error in learning path refresh task for course {course_id}: {str(e)}")
        # Re-raise the exception to mark the task as failed
        raise