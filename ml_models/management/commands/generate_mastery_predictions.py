from django.core.management.base import BaseCommand, CommandError
import logging
from ml_models.models import KnowledgeTracingModel, PredictionBatch
from ml_models.tasks import generate_mastery_predictions as generate_predictions_task
from core.models import Course

logger = logging.getLogger(__name__)

class Command(BaseCommand):
    help = 'Generate mastery predictions for all students in a course using a trained model'

    def add_arguments(self, parser):
        parser.add_argument('--model', type=str, required=True, help='Model type (dkt or sakt)')
        parser.add_argument('--course', type=str, required=True, help='Course ID')

    def handle(self, *args, **options):
        model_type = options['model']
        course_id = options['course']
        
        # Validate model type
        if model_type not in ['dkt', 'sakt']:
            raise CommandError(f"Invalid model type: {model_type}. Must be 'dkt' or 'sakt'.")
        
        # Get the course
        try:
            course = Course.objects.get(course_id=course_id)
        except Course.DoesNotExist:
            raise CommandError(f"Course {course_id} not found")
        
        # Get the model
        model = KnowledgeTracingModel.objects.filter(
            model_type=model_type,
            course=course,
            status='active'
        ).order_by('-created_at').first()
        
        if not model:
            raise CommandError(f"No active {model_type.upper()} model found for course {course_id}")
        
        # Create prediction batch
        batch = PredictionBatch.objects.create(
            model=model,
            status='pending'
        )
        
        self.stdout.write(f"Created prediction batch {batch.id} for model {model.name}")
        self.stdout.write("Generating predictions...")
        
        # Generate predictions
        generate_predictions_task(batch.id)
        
        self.stdout.write(self.style.SUCCESS(f"Successfully generated predictions for course {course_id}"))