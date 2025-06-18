from django.core.management.base import BaseCommand
from django.db import transaction
from django.utils import timezone
from datetime import timedelta
import random
from core.models import Student, Course


class Command(BaseCommand):
    help = 'Populate database with students and courses'

    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS('Starting to populate users and courses...'))
        
        with transaction.atomic():
            # Create course first
            course = self.create_course()
            
            # Create students
            students = self.create_students()
            
            # Enroll all students in the course
            self.enroll_students_in_course(course, students)
        
        self.stdout.write(self.style.SUCCESS('Successfully populated database with users and courses!'))

    def create_course(self):
        """Create the CS206 course"""
        course_id = "CS206"
        course_title = "CS206 - Intro to Java Programming"
        
        # Check if course already exists
        course, created = Course.objects.get_or_create(
            course_id=course_id,
            defaults={
                'title': course_title,
                'description': """
This introductory course provides students with a comprehensive foundation in Java programming. 
Students will learn core programming concepts including variables, data types, control structures, 
methods, arrays, and object-oriented programming principles. The course covers fundamental topics 
such as classes, objects, inheritance, polymorphism, and encapsulation. Students will develop 
problem-solving skills through hands-on programming exercises and projects. By the end of this 
course, students will be able to design, implement, and debug Java applications using industry 
best practices. Prerequisites: Basic understanding of computer systems and mathematical reasoning. 
This course serves as a foundation for advanced programming courses and software development.
                """.strip()
            }
        )
        
        if created:
            self.stdout.write(f'Created course: {course_title}')
        else:
            self.stdout.write(f'Course already exists: {course_title}')
        
        return course

    def create_students(self):
        """Create 10 realistic student users"""
        majors = ['Computer Science', 'Electrical Engineering', 'IT', 'Computer Engineering']
        academic_levels = ['freshman', 'sophomore', 'junior', 'senior']
        study_frequencies = ['daily', 'weekly', 'biweekly', 'monthly']
        
        students = []
        
        for i in range(10):
            # Generate student ID: A00 + 6 digits
            student_id = f"A00{str(i+1).zfill(6)}"
            
            # Check if student already exists
            if Student.objects.filter(student_id=student_id).exists():
                self.stdout.write(f'Student {student_id} already exists, skipping...')
                students.append(Student.objects.get(student_id=student_id))
                continue
            
            # Create Student with only required fields from the model
            student = Student.objects.create(
                student_id=student_id,
                major=random.choice(majors),
                academic_level=random.choice(academic_levels),
                gpa=round(random.uniform(2.5, 4.0), 2),
                prior_knowledge_score=round(random.uniform(0.3, 0.8), 2),
                study_frequency=random.choice(study_frequencies),
                attendance_rate=round(random.uniform(75.0, 98.0), 1),
                participation_score=round(random.uniform(60.0, 95.0), 1),
                last_login_date=timezone.now() - timedelta(days=random.randint(1, 30)),
                total_time_spent=timedelta(hours=random.randint(20, 100)),
                average_time_per_session=timedelta(minutes=random.randint(30, 120))
            )
            
            students.append(student)
            self.stdout.write(f'Created student: {student_id} ({student.major})')
        
        return students

    def enroll_students_in_course(self, course, students):
        """Enroll all students in the course"""
        enrolled_count = 0
        
        for student in students:
            if not course.students.filter(student_id=student.student_id).exists():
                course.students.add(student)
                enrolled_count += 1
                self.stdout.write(f'Enrolled {student.student_id} in {course.course_id}')
            else:
                self.stdout.write(f'Student {student.student_id} already enrolled in {course.course_id}')
        
        self.stdout.write(self.style.SUCCESS(f'Total students enrolled: {enrolled_count}'))
        self.stdout.write(f'Course {course.course_id} now has {course.students.count()} students')
