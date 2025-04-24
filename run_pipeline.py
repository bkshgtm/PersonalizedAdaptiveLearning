import os
import django
import logging
import argparse
from django.utils import timezone

# Set up Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'pal_project.settings')
django.setup()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from core.models import Course, Student, Topic
from ml_models.models import KnowledgeTracingModel, PredictionBatch
from knowledge_graph.models import KnowledgeGraph
from learning_paths.models import PathGenerator, PathGenerationJob
from generate_mastery_levels import generate_mastery_levels
from learning_paths.services.path_generator import LearningPathGenerator


def run_pipeline(course_id, model_type='dkt'):
    """
    Run the entire pipeline from model training to learning path generation.
    
    Args:
        course_id: ID of the course
        model_type: Type of model to use ('dkt' or 'sakt')
    """
    logger.info(f"Running pipeline for course {course_id} using {model_type.upper()} model")
    
    # Step 1: Generate mastery levels
    logger.info("Step 1: Generating mastery levels")
    generate_mastery_levels(course_id, model_type)
    
    # Step 2: Get the latest prediction batch
    logger.info("Step 2: Getting latest prediction batch")
    try:
        course = Course.objects.get(course_id=course_id)
        latest_batch = PredictionBatch.objects.filter(
            status='completed',
            model__course=course,
            model__model_type=model_type
        ).order_by('-completed_at').first()
        
        if not latest_batch:
            logger.error("No completed prediction batch found")
            return
        
        logger.info(f"Found prediction batch: {latest_batch.id}")
    except Course.DoesNotExist:
        logger.error(f"Course {course_id} not found")
        return
    
    # Step 3: Get the active knowledge graph
    logger.info("Step 3: Getting active knowledge graph")
    knowledge_graph = KnowledgeGraph.objects.filter(is_active=True).first()
    
    if not knowledge_graph:
        logger.error("No active knowledge graph found")
        return
    
    logger.info(f"Found knowledge graph: {knowledge_graph.name}")
    
    # Step 4: Get or create path generator
    logger.info("Step 4: Getting or creating path generator")
    admin_user = course.students.first().user if course.students.exists() else None
    
    if not admin_user:
        logger.error("No user found to create path generator")
        return
    
    path_generator, created = PathGenerator.objects.get_or_create(
        name=f"Default Generator for {course.title}",
        defaults={
            'description': 'Default path generator configuration',
            'created_by': admin_user,
            'is_active': True,
            'config': {
                'include_strong_topics': False,
                'max_resources_per_topic': 3
            }
        }
    )
    
    if created:
        logger.info(f"Created new path generator: {path_generator.name}")
    else:
        logger.info(f"Using existing path generator: {path_generator.name}")
    
    # Step 5: Generate learning paths for all students
    logger.info("Step 5: Generating learning paths for all students")
    students = Student.objects.filter(courses=course)
    
    if not students:
        logger.error("No students found for course")
        return
    
    logger.info(f"Found {students.count()} students")
    
    for i, student in enumerate(students):
        logger.info(f"Generating learning path for student {student.student_id} ({i+1}/{students.count()})")
        
        # Create path generation job
        job = PathGenerationJob.objects.create(
            generator=path_generator,
            student=student,
            course=course,
            prediction_batch=latest_batch,
            knowledge_graph=knowledge_graph,
            status='pending'
        )
        
        # Generate path
        try:
            path_generator_service = LearningPathGenerator(job.id)
            path = path_generator_service.generate_path()
            logger.info(f"Generated learning path with {path.items.count()} items")
        except Exception as e:
            logger.error(f"Error generating learning path: {str(e)}")
    
    logger.info("Pipeline completed successfully")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the PAL pipeline')
    parser.add_argument('--course', type=str, default='CS206', help='Course ID')
    parser.add_argument('--model', type=str, default='dkt', choices=['dkt', 'sakt'], help='Model type')
    
    args = parser.parse_args()
    
    run_pipeline(args.course, args.model)
