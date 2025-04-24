import os
import django

# Set up Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'pal_project.settings')
django.setup()

from django.contrib.auth.models import User
from learning_paths.services.path_generator import LearningPathGenerator
from learning_paths.models import PathGenerator, PathGenerationJob
from core.models import Student, Course
from ml_models.models import PredictionBatch

def test_path_generation():
    print("Testing learning path generation...")
    
    # Get test data
    student = Student.objects.first()
    course = Course.objects.first()
    latest_batch = PredictionBatch.objects.filter(status='completed').order_by('-completed_at').first()
    admin_user = User.objects.filter(is_superuser=True).first()

    if not student or not course or not admin_user:
        print("Error: Need at least one student, course, and admin user in database")
        return

    # Create or get path generator
    generator, _ = PathGenerator.objects.get_or_create(
        name="Default Generator",
        defaults={
            'description': 'Default path generator configuration',
            'created_by': admin_user,
            'config': {
                'include_strong_topics': False,
                'max_resources_per_topic': 3
            }
        }
    )

    # Create test job
    job = PathGenerationJob.objects.create(
        generator=generator,
        student=student,
        course=course,
        prediction_batch=latest_batch,
        status='pending'
    )

    # Generate path
    print(f"Generating path for student {student.student_id} in course {course.course_id}...")
    generator = LearningPathGenerator(job.id)
    path = generator.generate_path()

    # Print results
    print(f"\nGenerated path ID: {path.id}")
    print(f"Contains {path.items.count()} topics")
    print(f"Estimated completion time: {path.estimated_completion_time}")
    print("\nTop 5 topics:")
    for item in path.items.order_by('priority')[:5]:
        print(f"- {item.topic.name} (Priority: {item.priority}, Status: {item.status})")

if __name__ == '__main__':
    test_path_generation()
