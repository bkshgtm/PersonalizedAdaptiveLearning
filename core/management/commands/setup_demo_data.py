import logging
import datetime
import random
from django.core.management.base import BaseCommand
from django.contrib.auth.models import User
from django.utils import timezone
from django.db import transaction, models

from core.models import (
    Student, Course, Topic, Resource, Assessment, 
    Question, StudentInteraction
)

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = 'Populate the database with demo data for the PAL system'

    def add_arguments(self, parser):
        parser.add_argument(
            '--students',
            type=int,
            default=20,
            help='Number of students to create'
        )
        parser.add_argument(
            '--reset',
            action='store_true',
            help='Delete all existing data before creating new data'
        )

    def handle(self, *args, **options):
        num_students = options['students']
        reset = options['reset']
        
        if reset:
            self.reset_data()
            self.stdout.write(self.style.SUCCESS('Successfully reset all data'))
        
        try:
            with transaction.atomic():
                # Create an admin user if it doesn't exist
                self.create_admin_user()
                
                # Create courses
                courses = self.create_courses()
                self.stdout.write(self.style.SUCCESS(f'Created {len(courses)} courses'))
                
                # Create topics and resources
                topics, resources = self.create_topics_and_resources(courses)
                self.stdout.write(self.style.SUCCESS(f'Created {len(topics)} topics and {len(resources)} resources'))
                
                # Create assessments and questions
                assessments, questions = self.create_assessments_and_questions(courses, topics)
                self.stdout.write(self.style.SUCCESS(f'Created {len(assessments)} assessments and {len(questions)} questions'))
                
                # Create students
                students = self.create_students(num_students, courses)
                self.stdout.write(self.style.SUCCESS(f'Created {len(students)} students'))
                
                # Create student interactions
                interactions = self.create_interactions(students, questions)
                self.stdout.write(self.style.SUCCESS(f'Created {len(interactions)} student interactions'))
                
            self.stdout.write(self.style.SUCCESS('Successfully populated database with demo data'))
            
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'Error setting up demo data: {str(e)}'))
            logger.exception("Error in setup_demo_data command")
            raise

    def reset_data(self):
        """Delete all existing data"""
        StudentInteraction.objects.all().delete()
        Question.objects.all().delete()
        Assessment.objects.all().delete()
        Resource.objects.all().delete()
        Topic.objects.all().delete()
        Student.objects.all().delete()
        Course.objects.all().delete()

    def create_admin_user(self):
        """Create an admin user if it doesn't exist"""
        if not User.objects.filter(username='admin').exists():
            User.objects.create_superuser(
                username='admin',
                email='admin@example.com',
                password='adminpassword'
            )
            self.stdout.write(self.style.SUCCESS('Created admin user'))

    def create_courses(self):
        """Create Java programming courses"""
        courses = []
        
        # Main Java course
        java_course = Course.objects.create(
            course_id='JAVA101',
            title='Introduction to Java Programming',
            description='A comprehensive introduction to Java programming language and object-oriented concepts.'
        )
        courses.append(java_course)
        
        # Advanced Java course
        adv_java = Course.objects.create(
            course_id='JAVA201',
            title='Advanced Java Programming',
            description='Advanced topics in Java including multithreading, networking, and design patterns.'
        )
        courses.append(adv_java)
        
        return courses

    def create_topics_and_resources(self, courses):
        """Create topics and resources for the courses"""
        topics = []
        resources = []
        
        java_course = courses[0]
        
        # Topics for Introduction to Java
        java_topics = [
            # Core concepts
            {
                'name': 'Java Basics',
                'description': 'Introduction to Java syntax, structure, and environment',
                'subtopics': [
                    {'name': 'Java Environment Setup', 'description': 'Setting up Java development environment'},
                    {'name': 'Hello World Program', 'description': 'First Java program'},
                    {'name': 'Java Syntax', 'description': 'Basic syntax rules in Java'}
                ]
            },
            {
                'name': 'Variables and Data Types',
                'description': 'Java variables, primitive types, and reference types',
                'subtopics': [
                    {'name': 'Primitive Types', 'description': 'int, float, double, char, boolean, etc.'},
                    {'name': 'Reference Types', 'description': 'Strings and objects'},
                    {'name': 'Type Conversion', 'description': 'Converting between data types'}
                ]
            },
            {
                'name': 'Operators',
                'description': 'Java operators for arithmetic, comparison, and logical operations',
                'subtopics': [
                    {'name': 'Arithmetic Operators', 'description': '+, -, *, /, %'},
                    {'name': 'Comparison Operators', 'description': '==, !=, >, <, >=, <='},
                    {'name': 'Logical Operators', 'description': '&&, ||, !'}
                ]
            },
            {
                'name': 'Control Flow',
                'description': 'Controlling program flow with conditionals and loops',
                'subtopics': [
                    {'name': 'Conditional Statements', 'description': 'if, else, switch'},
                    {'name': 'For Loops', 'description': 'for, enhanced for loop'},
                    {'name': 'While Loops', 'description': 'while, do-while'}
                ]
            },
            {
                'name': 'Arrays',
                'description': 'Working with arrays in Java',
                'subtopics': [
                    {'name': 'Array Declaration', 'description': 'Creating and initializing arrays'},
                    {'name': 'Array Operations', 'description': 'Accessing and modifying array elements'},
                    {'name': 'Multidimensional Arrays', 'description': '2D and multidimensional arrays'}
                ]
            },
            {
                'name': 'Methods',
                'description': 'Creating and using methods in Java',
                'subtopics': [
                    {'name': 'Method Declaration', 'description': 'Defining methods'},
                    {'name': 'Method Parameters', 'description': 'Passing data to methods'},
                    {'name': 'Return Types', 'description': 'Returning values from methods'}
                ]
            },
            {
                'name': 'Object-Oriented Programming',
                'description': 'OOP concepts in Java',
                'subtopics': [
                    {'name': 'Classes and Objects', 'description': 'Creating classes and objects'},
                    {'name': 'Inheritance', 'description': 'Extending classes'},
                    {'name': 'Polymorphism', 'description': 'Method overriding and overloading'},
                    {'name': 'Encapsulation', 'description': 'Access modifiers and data hiding'},
                    {'name': 'Abstraction', 'description': 'Abstract classes and interfaces'}
                ]
            },
            {
                'name': 'Exception Handling',
                'description': 'Handling exceptions in Java',
                'subtopics': [
                    {'name': 'Try-Catch Blocks', 'description': 'Catching and handling exceptions'},
                    {'name': 'Throwing Exceptions', 'description': 'Creating and throwing exceptions'},
                    {'name': 'Custom Exceptions', 'description': 'Creating custom exception classes'}
                ]
            },
            {
                'name': 'Collections',
                'description': 'Java Collections Framework',
                'subtopics': [
                    {'name': 'Lists', 'description': 'ArrayList, LinkedList'},
                    {'name': 'Sets', 'description': 'HashSet, TreeSet'},
                    {'name': 'Maps', 'description': 'HashMap, TreeMap'},
                    {'name': 'Iterators', 'description': 'Traversing collections'}
                ]
            }
        ]
        
        # Create topics
        for topic_data in java_topics:
            main_topic = Topic.objects.create(
                name=topic_data['name'],
                description=topic_data['description'],
                course=java_course
            )
            topics.append(main_topic)
            
            # Create subtopics
            for subtopic_data in topic_data.get('subtopics', []):
                subtopic = Topic.objects.create(
                    name=subtopic_data['name'],
                    description=subtopic_data['description'],
                    course=java_course,
                    parent=main_topic
                )
                topics.append(subtopic)
                
                # Create resources for subtopic
                resources.extend(self.create_resources_for_topic(subtopic))
            
            # Create resources for main topic
            resources.extend(self.create_resources_for_topic(main_topic))
        
        return topics, resources

    def create_resources_for_topic(self, topic):
        """Create resources for a topic"""
        resources = []
        
        # Resource types and difficulties
        resource_types = ['video', 'document', 'exercise', 'tutorial', 'quiz']
        difficulties = ['beginner', 'intermediate', 'advanced']
        
        # Create 2-4 resources per topic
        num_resources = random.randint(2, 4)
        
        for i in range(num_resources):
            resource_type = random.choice(resource_types)
            difficulty = random.choice(difficulties)
            
            # Estimated time (10-60 minutes)
            estimated_time = datetime.timedelta(minutes=random.randint(10, 60))
            
            # Create resource
            resource = Resource.objects.create(
                title=f"{topic.name} {resource_type.capitalize()} {i+1}",
                description=f"A {difficulty} level {resource_type} for {topic.name}",
                url=f"https://example.com/resources/{topic.name.lower().replace(' ', '-')}/{resource_type}/{i+1}",
                resource_type=resource_type,
                difficulty=difficulty,
                estimated_time=estimated_time
            )
            
            # Add topic to resource
            resource.topics.add(topic)
            
            resources.append(resource)
        
        return resources

    def create_assessments_and_questions(self, courses, topics):
        """Create assessments and questions for the courses"""
        assessments = []
        questions = []
        
        java_course = courses[0]
        
        # Assessment types
        assessment_types = ['quiz', 'exam', 'assignment', 'project']
        
        # Question types
        question_types = ['mcq', 'coding', 'fill_blank', 'short_answer']
        
        # Create 5 assessments
        for i in range(5):
            assessment_type = assessment_types[i % len(assessment_types)]
            
            # Create assessment
            assessment = Assessment.objects.create(
                assessment_id=f"JAVA101-{assessment_type.upper()}-{i+1}",
                title=f"Java {assessment_type.capitalize()} {i+1}",
                assessment_type=assessment_type,
                course=java_course,
                date=timezone.now() - datetime.timedelta(days=30-i*7),
                proctored=(assessment_type == 'exam')
            )
            
            assessments.append(assessment)
            
            # Create 5-10 questions per assessment
            num_questions = random.randint(5, 10)
            
            for j in range(num_questions):
                question_type = question_types[j % len(question_types)]
                
                # Select a random topic for this question
                topic = random.choice(topics)
                
                # Create question
                question = Question.objects.create(
                    question_id=f"{assessment.assessment_id}-Q{j+1}",
                    assessment=assessment,
                    text=f"Question {j+1} for {topic.name}",
                    question_type=question_type,
                    topic=topic
                )
                
                questions.append(question)
        
        return assessments, questions

    def create_students(self, num_students, courses):
        """Create students and enroll them in courses"""
        students = []
        
        # Student data
        majors = ['Computer Science', 'Information Technology', 'Software Engineering', 'Data Science', 'Cybersecurity']
        academic_levels = ['freshman', 'sophomore', 'junior', 'senior']
        study_frequencies = ['daily', 'weekly', 'biweekly', 'monthly']
        
        for i in range(num_students):
            student_id = f"S{100 + i:03d}"
            major = random.choice(majors)
            academic_level = random.choice(academic_levels)
            gpa = round(random.uniform(2.0, 4.0), 2)
            prior_knowledge_score = round(random.uniform(0.0, 1.0), 2)
            study_frequency = random.choice(study_frequencies)
            attendance_rate = round(random.uniform(60.0, 100.0), 1)
            participation_score = round(random.uniform(0.0, 10.0), 1)
            
            # Create student
            student = Student.objects.create(
                student_id=student_id,
                major=major,
                academic_level=academic_level,
                gpa=gpa,
                prior_knowledge_score=prior_knowledge_score,
                study_frequency=study_frequency,
                attendance_rate=attendance_rate,
                participation_score=participation_score,
                last_login_date=timezone.now() - datetime.timedelta(days=random.randint(0, 14)),
                total_time_spent=datetime.timedelta(hours=random.randint(10, 100)),
                average_time_per_session=datetime.timedelta(minutes=random.randint(30, 120))
            )
            
            # Enroll in courses
            for course in courses:
                # 80% chance of enrolling in each course
                if random.random() < 0.8:
                    student.courses.add(course)
            
            students.append(student)
        
        return students

    def create_interactions(self, students, questions):
        """Create student interactions with questions"""
        interactions = []
        
        # For each student, create interactions with some questions
        for student in students:
            # Get courses the student is enrolled in
            student_courses = student.courses.all()
            if not student_courses:
                continue
                
            # Get questions from student's courses
            course_questions = [
                q for q in questions
                if q.assessment.course in student_courses
            ]
            if not course_questions:
                continue
                
            # Create 10-50 interactions per student
            num_interactions = random.randint(10, 50)
            for _ in range(num_interactions):
                # Select a random question
                question = random.choice(course_questions)
                
                # Determine correctness (better students have higher chance of being correct)
                correct = random.random() < (0.5 + student.gpa / 8.0)
                
                # Score (0-100)
                score = random.randint(70, 100) if correct else random.randint(0, 69)
                
                # Time taken (1-20 minutes)
                time_taken = datetime.timedelta(minutes=random.randint(1, 20))
                
                # Timestamp (within the last 60 days)
                timestamp = timezone.now() - datetime.timedelta(days=random.randint(0, 60))
                
                # Find next available attempt number using aggregate
                max_attempt = StudentInteraction.objects.filter(
                    student=student,
                    question=question
                ).aggregate(models.Max('attempt_number'))['attempt_number__max'] or 0
                
                # Use the next attempt number
                attempt_number = max_attempt + 1
                
                # Resource viewed before
                resource_viewed_before = random.random() < 0.3
                
                # Create interaction
                interaction = StudentInteraction.objects.create(
                    student=student,
                    question=question,
                    response=f"Student response for question {question.question_id}",
                    correct=correct,
                    score=score,
                    time_taken=time_taken,
                    timestamp=timestamp,
                    attempt_number=attempt_number,
                    resource_viewed_before=resource_viewed_before
                )
                interactions.append(interaction)
                
        return interactions
