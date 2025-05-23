import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'pal_project.settings')
django.setup()

from learning_paths.models import LearningPath, PathGenerationJob
from django.db import transaction

def clear_learning_paths():
    with transaction.atomic():
        path_count = LearningPath.objects.count()
        job_count = PathGenerationJob.objects.count()
        
        LearningPath.objects.all().delete()
        
        PathGenerationJob.objects.all().delete()
        
        print(f"Successfully deleted {path_count} learning paths and {job_count} generation jobs")

if __name__ == "__main__":
    clear_learning_paths()