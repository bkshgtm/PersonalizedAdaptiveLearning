# topic_classification/management/commands/setup_dissect_classifier.py

import logging
import os
from django.core.management.base import BaseCommand
from django.contrib.auth.models import User
from django.conf import settings

from topic_classification.models import ClassificationModel
from core.models import Course

logger = logging.getLogger(__name__)

class Command(BaseCommand):
    help = "Sets up Dissect as the default classifier for topic classification"

    def handle(self, *args, **options):
        self.stdout.write("Setting up Dissect classifier as default...")

        # Ensure we have an admin user
        if not User.objects.filter(is_staff=True).exists():
            self.stdout.write("Creating admin user...")
            User.objects.create_superuser(
                username='admin',
                email='admin@example.com',
                password='adminpassword'
            )
        
        admin_user = User.objects.filter(is_staff=True).first()
        
        # Get courses
        courses = Course.objects.all()
        if not courses.exists():
            self.stdout.write("No courses found. Please create a course first.")
            return
        
        # Create model path directory if it doesn't exist
        model_dir = os.path.join(settings.MODEL_STORAGE_PATH, 'topic_classification')
        os.makedirs(model_dir, exist_ok=True)
        
        # For each course, set up a Dissect classifier
        for course in courses:
            self.stdout.write(f"Setting up Dissect classifier for course: {course.title}")

            # Unset current default models for this course
            ClassificationModel.objects.filter(
                is_default=True
            ).update(is_default=False)

            # Check if a Dissect model already exists
            existing_model = ClassificationModel.objects.filter(
                model_type='dissect'
            ).first()

            if existing_model:
                self.stdout.write(f"Dissect classifier already exists: {existing_model.name}")
                existing_model.is_default = True
                existing_model.save()
                continue

            # Create a new Dissect model
            model_path = os.path.join(model_dir, f'dissect_{course.course_id}.json')

            model = ClassificationModel.objects.create(
                name=f"Dissect Classifier for {course.title}",
                model_type='dissect',
                description="Dissect API-based topic classifier",
                created_by=admin_user,
                model_path=model_path,
                is_default=True,
                status='active',  # DeepSeek models are active from the start
                metadata={
                    'api_config': {
                        'model': 'dissect-coder-1.3b-instruct', # Assuming similar model naming
                        'temperature': 0.1
                    }
                }
            )

            self.stdout.write(f"Created Dissect classifier: {model.name}")

        self.stdout.write(self.style.SUCCESS("Dissect classifier setup complete!"))
