import random
import datetime
from django.core.management.base import BaseCommand
from django.utils import timezone
from django.db import transaction

from core.models import Student, Course


class Command(BaseCommand):
    help = 'Delete all students and create new ones with specific ID pattern'

    def add_arguments(self, parser):
        parser.add_argument(
            '--course_id',
            type=str,
            default='CS206',
            help='Course ID to enroll students in'
        )
        parser.add_argument(
            '--num_students',
            type=int,
            default=10,
            help='Number of students to create'
        )
        parser.add_argument(
            '--id_prefix',
            type=str,
            default='A0058',
            help='Student ID prefix'
        )

    def handle(self, *args, **options):
        course_id = options['course_id']
        num_students = options['num_students']
        id_prefix = options['id_prefix']

        try:
            # Load the course
            course = Course.objects.get(course_id=course_id)
            self.stdout.write(f"Found course: {course.title}")
            
            with transaction.atomic():
                # Delete all existing students
                deleted_count, _ = Student.objects.all().delete()
                self.stdout.write(f"Deleted {deleted_count} existing students")
                
                # Create new students
                self.create_students(course, num_students, id_prefix)
                
            self.stdout.write(self.style.SUCCESS(f"Successfully reset students database"))
            
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Error: {str(e)}"))
    
    def create_students(self, course, num_students, id_prefix):
        """Create students with specific ID pattern and enroll them in the course"""
        majors = ['Computer Science', 'Information Technology', 'Software Engineering', 
                  'Data Science', 'Cybersecurity', 'Digital Media', 'Game Development']
        academic_levels = ['freshman', 'sophomore', 'junior', 'senior']
        study_frequencies = ['daily', 'weekly', 'biweekly', 'monthly']
        
        # Create student profiles with different learning characteristics
        student_profiles = [
            # Visual learners
            {'learning_style': 'visual', 'gpa_range': (2.8, 4.0), 'prior_knowledge': (0.3, 0.8)},
            # Practical learners
            {'learning_style': 'practical', 'gpa_range': (2.5, 3.8), 'prior_knowledge': (0.2, 0.7)},
            # Theoretical learners
            {'learning_style': 'theoretical', 'gpa_range': (3.0, 4.0), 'prior_knowledge': (0.4, 0.9)},
            # Struggling learners
            {'learning_style': 'struggling', 'gpa_range': (2.0, 3.0), 'prior_knowledge': (0.1, 0.4)},
            # Advanced learners
            {'learning_style': 'advanced', 'gpa_range': (3.5, 4.0), 'prior_knowledge': (0.7, 1.0)}
        ]
        
        students_created = 0
        
        for i in range(num_students):
            # Select a profile for this student
            profile = random.choice(student_profiles)
            
            # Generate student ID
            student_id = f"{id_prefix}{random.randint(1000, 9999)}"
            
            # Generate student attributes based on profile
            major = random.choice(majors)
            academic_level = random.choice(academic_levels)
            gpa = round(random.uniform(*profile['gpa_range']), 2)
            prior_knowledge = round(random.uniform(*profile['prior_knowledge']), 2)
            study_frequency = random.choice(study_frequencies)
            
            # Attendance and participation correlate with profile
            if profile['learning_style'] in ['theoretical', 'advanced']:
                attendance = random.uniform(75, 98)
                participation = random.uniform(7, 10)
            elif profile['learning_style'] in ['visual', 'practical']:
                attendance = random.uniform(65, 90)
                participation = random.uniform(5, 9)
            else:  # struggling
                attendance = random.uniform(50, 80)
                participation = random.uniform(2, 7)
            
            # Last login and time spent
            last_login_delta = random.randint(0, 14)
            total_hours = random.randint(10, 100)
            
            # Create student
            student = Student.objects.create(
                student_id=student_id,
                major=major,
                academic_level=academic_level,
                gpa=gpa,
                prior_knowledge_score=prior_knowledge,
                study_frequency=study_frequency,
                attendance_rate=round(attendance, 1),
                participation_score=round(participation, 1),
                last_login_date=timezone.now() - datetime.timedelta(days=last_login_delta),
                total_time_spent=datetime.timedelta(hours=total_hours),
                average_time_per_session=datetime.timedelta(minutes=random.randint(30, 120))
            )
            
            # Enroll in the course
            student.courses.add(course)
            students_created += 1
            
            self.stdout.write(f"Created student {student_id} ({profile['learning_style']} learner)")
        
        self.stdout.write(self.style.SUCCESS(f"Successfully created {students_created} students"))