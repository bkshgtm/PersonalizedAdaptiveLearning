import logging
import datetime
import random
import math
from django.core.management.base import BaseCommand
from django.contrib.auth.models import User
from django.utils import timezone
from django.db import transaction

from core.models import (
    Student, Course, Topic, Resource, Assessment, 
    Question, StudentInteraction
)

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = 'Populate the database with meaningful data for ML-based learning path generation'

    def add_arguments(self, parser):
        parser.add_argument(
            '--students',
            type=int,
            default=100,
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
                
                # Create Java course
                java_course = Course.objects.create(
                    course_id='CS206',
                    title='Introduction to Java Programming',
                    description='A comprehensive introduction to Java programming language and object-oriented concepts.'
                )
                self.stdout.write(self.style.SUCCESS('Created Introduction to Java Programming course'))
                
                # Create topics with meaningful structure
                topics, topic_prereqs = self.create_structured_topics(java_course)
                self.stdout.write(self.style.SUCCESS(f'Created {len(topics)} structured topics'))
                
                # Create resources with effectiveness metadata
                resources = self.create_meaningful_resources(topics)
                self.stdout.write(self.style.SUCCESS(f'Created {len(resources)} resources with effectiveness data'))
                
                # Create assessments and questions with concept links
                assessments, questions = self.create_concept_linked_assessments(java_course, topics)
                self.stdout.write(self.style.SUCCESS(f'Created {len(assessments)} assessments and {len(questions)} questions'))
                
                # Create realistic student profiles
                students = self.create_student_profiles(num_students, java_course)
                self.stdout.write(self.style.SUCCESS(f'Created {len(students)} student profiles'))
                
                # Create learning progression interactions
                interactions = self.create_learning_progression_data(students, questions, resources, topic_prereqs)
                self.stdout.write(self.style.SUCCESS(f'Created {len(interactions)} learning progression interactions'))
                
            self.stdout.write(self.style.SUCCESS('Successfully populated database with ML-ready educational data'))
            
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'Error setting up ML-ready data: {str(e)}'))
            logger.exception("Error in ml_data_setup command")
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

    def create_structured_topics(self, course):
        """Create topics with clear learning structure and metadata"""
        topics = {}
        topic_prereqs = {}
        
        # Define Java topics with clear difficulty progression
        java_topics = [
            {
                'name': 'Java Basics',
                'description': 'Introduction to Java syntax, structure, and environment',
                'difficulty': 1,
                'importance': 5,
                'prerequisites': []
            },
            {
                'name': 'Variables and Data Types',
                'description': 'Java variables, primitive types, and reference types',
                'difficulty': 2,
                'importance': 5,
                'prerequisites': ['Java Basics']
            },
            {
                'name': 'Operators',
                'description': 'Java operators for arithmetic, comparison, and logical operations',
                'difficulty': 2,
                'importance': 4,
                'prerequisites': ['Variables and Data Types']
            },
            {
                'name': 'Control Flow',
                'description': 'Controlling program flow with conditionals and loops',
                'difficulty': 3,
                'importance': 5,
                'prerequisites': ['Operators']
            },
            {
                'name': 'Methods',
                'description': 'Creating and using methods in Java',
                'difficulty': 3,
                'importance': 5,
                'prerequisites': ['Control Flow']
            },
            {
                'name': 'Arrays',
                'description': 'Working with arrays in Java',
                'difficulty': 4,
                'importance': 4,
                'prerequisites': ['Control Flow']
            },
            {
                'name': 'Object-Oriented Fundamentals',
                'description': 'Basic OOP concepts in Java',
                'difficulty': 4,
                'importance': 5,
                'prerequisites': ['Methods', 'Arrays']
            },
            {
                'name': 'Classes and Objects',
                'description': 'Creating and using classes and objects',
                'difficulty': 4,
                'importance': 5,
                'prerequisites': ['Object-Oriented Fundamentals']
            },
            {
                'name': 'Inheritance',
                'description': 'Class inheritance and method overriding',
                'difficulty': 5,
                'importance': 4,
                'prerequisites': ['Classes and Objects']
            },
            {
                'name': 'Polymorphism',
                'description': 'Polymorphic behavior in Java',
                'difficulty': 5,
                'importance': 4,
                'prerequisites': ['Inheritance']
            },
            {
                'name': 'Encapsulation',
                'description': 'Data hiding and access control',
                'difficulty': 4,
                'importance': 4,
                'prerequisites': ['Classes and Objects']
            },
            {
                'name': 'Interfaces',
                'description': 'Creating and implementing interfaces',
                'difficulty': 5,
                'importance': 4,
                'prerequisites': ['Inheritance']
            },
            {
                'name': 'Exception Handling',
                'description': 'Managing exceptions in Java',
                'difficulty': 4,
                'importance': 4,
                'prerequisites': ['Methods', 'Classes and Objects']
            },
            {
                'name': 'Collections Framework',
                'description': 'Using Java collections classes',
                'difficulty': 5,
                'importance': 4,
                'prerequisites': ['Object-Oriented Fundamentals', 'Arrays']
            },
            {
                'name': 'File I/O',
                'description': 'Reading and writing files in Java',
                'difficulty': 4,
                'importance': 3,
                'prerequisites': ['Exception Handling']
            }
        ]
        
        # Create all topics first
        for topic_data in java_topics:
            metadata = {
                'difficulty_level': topic_data['difficulty'],
                'importance_score': topic_data['importance'],
                'estimated_study_time': f"{topic_data['difficulty'] * 2}h"
            }
            
            # Convert metadata to JSON string (assuming Topic model has a JSONField for metadata)
            # If not, adapt the code according to your model's structure
            topic = Topic.objects.create(
                name=topic_data['name'],
                description=topic_data['description'],
                course=course,
                # Store metadata as appropriate for your model structure
                # If you have specific fields, use them directly
                # If you have a metadata field, convert to string
                # metadata=json.dumps(metadata)
            )
            
            topics[topic_data['name']] = topic
            topic_prereqs[topic_data['name']] = topic_data['prerequisites']
        
        # Create subtopics and connections
        for main_topic_name, subtopic_names in {
            'Variables and Data Types': ['Primitive Types', 'Reference Types', 'Type Conversion'],
            'Control Flow': ['Conditional Statements', 'For Loops', 'While Loops'],
            'Methods': ['Method Declaration', 'Parameters and Arguments', 'Return Values'],
            'Object-Oriented Fundamentals': ['Object Basics', 'Class Structure', 'Constructors']
        }.items():
            if main_topic_name in topics:
                main_topic = topics[main_topic_name]
                for i, subtopic_name in enumerate(subtopic_names):
                    subtopic = Topic.objects.create(
                        name=subtopic_name,
                        description=f"Subtopic of {main_topic_name}",
                        course=course,
                        parent=main_topic
                    )
                    topics[subtopic_name] = subtopic
        
        return topics, topic_prereqs

    def create_meaningful_resources(self, topics):
        """Create resources with meaningful effectiveness data"""
        resources = []
        resource_types = ['video', 'document', 'exercise', 'tutorial', 'quiz']
        learning_styles = ['visual', 'auditory', 'reading', 'kinesthetic']
        
        # Define resource effectiveness patterns
        # Each topic will have resources with different effectiveness for different learning styles
        for topic_name, topic in topics.items():
            # Create 3-5 resources per topic with strategic effectiveness patterns
            num_resources = random.randint(3, 5)
            
            for i in range(num_resources):
                resource_type = resource_types[i % len(resource_types)]
                
                # Create effectiveness scores for different learning styles
                # Some resources are better for visual learners, others for kinesthetic, etc.
                effectiveness = {}
                primary_style = learning_styles[i % len(learning_styles)]
                for style in learning_styles:
                    if style == primary_style:
                        effectiveness[style] = random.uniform(0.7, 0.9)  # High for primary style
                    else:
                        effectiveness[style] = random.uniform(0.3, 0.6)  # Lower for other styles
                
                # Estimated time (appropriate for topic difficulty)
                topic_difficulty = getattr(topic, 'difficulty_level', 3)
                if not isinstance(topic_difficulty, (int, float)):
                    topic_difficulty = 3
                    
                base_minutes = 10 + (topic_difficulty * 5)
                estimated_time = datetime.timedelta(minutes=random.randint(
                    int(base_minutes * 0.8), 
                    int(base_minutes * 1.2)
                ))
                
                # Create resource with meaningful metadata
                title = f"{topic_name} {resource_type.capitalize()} {i+1}"
                description = f"A resource for learning {topic_name} through {resource_type}"
                
                # Additional metadata based on resource type
                if resource_type == 'video':
                    description += f" (Visual demonstration with examples)"
                elif resource_type == 'exercise':
                    description += f" (Hands-on practice)"
                elif resource_type == 'tutorial':
                    description += f" (Step-by-step guide)"
                    
                resource = Resource.objects.create(
                    title=title,
                    description=description,
                    url=f"https://example.com/resources/{topic_name.lower().replace(' ', '-')}/{resource_type}/{i+1}",
                    resource_type=resource_type,
                    difficulty='beginner' if topic_difficulty <= 2 else 'intermediate' if topic_difficulty <= 4 else 'advanced',
                    estimated_time=estimated_time,
                    # Add effectiveness metadata as appropriate for your model
                    # If you have a specific field: effectiveness_data=json.dumps(effectiveness)
                )
                
                # Add topic to resource
                resource.topics.add(topic)
                resources.append(resource)
        
        return resources

    def create_concept_linked_assessments(self, course, topics):
        """Create assessments and questions with explicit concept linkage"""
        assessments = []
        questions = []
        
        # Define different types of questions that test different cognitive levels
        question_types = {
            'knowledge': ['mcq', 'fill_blank'],
            'comprehension': ['mcq', 'short_answer'],
            'application': ['coding', 'short_answer'],
            'analysis': ['coding', 'long_answer']
        }
        
        # Create formative assessments (quizzes) for each major topic
        for topic_name, topic in topics.items():
            if '.' not in topic_name:  # Skip subtopics
                # Create quiz for this topic
                assessment = Assessment.objects.create(
                    assessment_id=f"JAVA-{topic_name.replace(' ', '-').upper()}-QUIZ",
                    title=f"{topic_name} Quiz",
                    assessment_type='quiz',
                    course=course,
                    date=timezone.now() - datetime.timedelta(days=random.randint(15, 60)),
                    proctored=False
                )
                assessments.append(assessment)
                
                # Add 5-8 questions of increasing difficulty and cognitive levels
                num_questions = random.randint(5, 8)
                for i in range(num_questions):
                    # Pick cognitive level based on question index
                    if i < num_questions * 0.4:
                        cognitive_level = 'knowledge'
                    elif i < num_questions * 0.7:
                        cognitive_level = 'comprehension'
                    elif i < num_questions * 0.9:
                        cognitive_level = 'application'
                    else:
                        cognitive_level = 'analysis'
                    
                    # Select question type appropriate for cognitive level
                    question_type = random.choice(question_types[cognitive_level])
                    
                    # Create question with explicit concept linkage
                    question = Question.objects.create(
                        question_id=f"{assessment.assessment_id}-Q{i+1}",
                        assessment=assessment,
                        text=f"{cognitive_level.capitalize()} question {i+1} on {topic_name}",
                        question_type=question_type,
                        topic=topic,
                        # Add metadata indicating cognitive level and difficulty
                        # If you have specific fields or metadata field: 
                        # metadata=json.dumps({'cognitive_level': cognitive_level, 'difficulty': float(i+1)/num_questions})
                    )
                    questions.append(question)
        
        # Create 3 comprehensive exams covering multiple topics
        for i in range(3):
            if i == 0:
                exam_title = "Java Basics Exam"
                exam_topics = [t for n, t in topics.items() if n in ['Java Basics', 'Variables and Data Types', 'Operators', 'Control Flow']]
                difficulty = 'beginner'
            elif i == 1:
                exam_title = "Java Methods and Arrays Exam"
                exam_topics = [t for n, t in topics.items() if n in ['Methods', 'Arrays', 'Control Flow']]
                difficulty = 'intermediate'
            else:
                exam_title = "Java OOP Concepts Exam"
                exam_topics = [t for n, t in topics.items() if n in ['Object-Oriented Fundamentals', 'Classes and Objects', 'Inheritance', 'Polymorphism']]
                difficulty = 'advanced'
            
            # Create exam
            assessment = Assessment.objects.create(
                assessment_id=f"JAVA-{exam_title.replace(' ', '-').upper()}",
                title=exam_title,
                assessment_type='exam',
                course=course,
                date=timezone.now() - datetime.timedelta(days=random.randint(5, 30)),
                proctored=True
            )
            assessments.append(assessment)
            
            # Add questions covering different topics and difficulty levels
            num_questions = random.randint(10, 15)
            for i in range(num_questions):
                # Select topic
                topic = random.choice(exam_topics)
                
                # Determine cognitive level based on question index
                if i < num_questions * 0.3:
                    cognitive_level = 'knowledge'
                elif i < num_questions * 0.6:
                    cognitive_level = 'comprehension'
                elif i < num_questions * 0.9:
                    cognitive_level = 'application'
                else:
                    cognitive_level = 'analysis'
                
                question_type = random.choice(question_types[cognitive_level])
                
                # Create question
                question = Question.objects.create(
                    question_id=f"{assessment.assessment_id}-Q{i+1}",
                    assessment=assessment,
                    text=f"{cognitive_level.capitalize()} question {i+1} on {topic.name}",
                    question_type=question_type,
                    topic=topic,
                    # Additional metadata as appropriate
                )
                questions.append(question)
        
        return assessments, questions

    def create_student_profiles(self, num_students, course):
        """Create students with meaningful learning profiles"""
        students = []
        
        # Define various profile templates that reflect different learning styles and abilities
        # These will create meaningful patterns in the data
        student_profiles = [
            # Visual learners - strong on conceptual, weaker on details
            {
                'profile': 'visual_learner', 
                'weight': 0.25,
                'attributes': {
                    'gpa_range': (2.8, 4.0),
                    'prior_knowledge_range': (0.3, 0.8),
                    'study_frequencies': ['daily', 'weekly'],
                    'attendance_range': (70, 95),
                    'participation_range': (5, 10),
                    'learning_style': 'visual',
                    'strengths': ['conceptual_understanding', 'pattern_recognition'],
                    'weaknesses': ['detail_orientation', 'memorization']
                }
            },
            # Practical learners - strong on application, weaker on theory
            {
                'profile': 'practical_learner',
                'weight': 0.25,
                'attributes': {
                    'gpa_range': (2.5, 3.8),
                    'prior_knowledge_range': (0.2, 0.7),
                    'study_frequencies': ['weekly', 'biweekly'],
                    'attendance_range': (60, 90),
                    'participation_range': (6, 9),
                    'learning_style': 'kinesthetic',
                    'strengths': ['problem_solving', 'application'],
                    'weaknesses': ['theoretical_understanding', 'abstract_concepts']
                }
            },
            # Theoretical learners - strong on concepts, weaker on application
            {
                'profile': 'theoretical_learner',
                'weight': 0.2,
                'attributes': {
                    'gpa_range': (3.0, 4.0),
                    'prior_knowledge_range': (0.4, 0.9),
                    'study_frequencies': ['daily', 'weekly'],
                    'attendance_range': (75, 98),
                    'participation_range': (7, 10),
                    'learning_style': 'reading',
                    'strengths': ['abstract_reasoning', 'theoretical_understanding'],
                    'weaknesses': ['practical_application', 'time_management']
                }
            },
            # Struggling learners - need more structured help
            {
                'profile': 'struggling_learner',
                'weight': 0.15,
                'attributes': {
                    'gpa_range': (2.0, 3.0),
                    'prior_knowledge_range': (0.1, 0.4),
                    'study_frequencies': ['weekly', 'biweekly', 'monthly'],
                    'attendance_range': (50, 80),
                    'participation_range': (2, 7),
                    'learning_style': 'mixed',
                    'strengths': ['persistence', 'seeking_help'],
                    'weaknesses': ['foundational_knowledge', 'self_direction']
                }
            },
            # Advanced learners - already know much of the material
            {
                'profile': 'advanced_learner',
                'weight': 0.15,
                'attributes': {
                    'gpa_range': (3.5, 4.0),
                    'prior_knowledge_range': (0.7, 1.0),
                    'study_frequencies': ['weekly', 'biweekly'],
                    'attendance_range': (60, 90),  # May skip classes they know well
                    'participation_range': (8, 10),
                    'learning_style': 'self_directed',
                    'strengths': ['independent_learning', 'advanced_concepts'],
                    'weaknesses': ['patience_with_basics', 'collaborative_learning']
                }
            }
        ]
        
        # Create students based on weighted profiles
        profile_weights = [p['weight'] for p in student_profiles]
        total_weight = sum(profile_weights)
        normalized_weights = [w/total_weight for w in profile_weights]
        
        majors = ['Computer Science', 'Information Technology', 'Software Engineering', 
                  'Data Science', 'Cybersecurity', 'Digital Media', 'Game Development']
        academic_levels = ['freshman', 'sophomore', 'junior', 'senior']
        
        for i in range(num_students):
            # Select profile based on weights
            profile = random.choices(student_profiles, normalized_weights)[0]['attributes']
            
            # Generate student data based on profile
            student_id = f"A0058{random.randint(0, 9999):04d}"
            major = random.choice(majors)
            academic_level = random.choice(academic_levels)
            gpa = round(random.uniform(*profile['gpa_range']), 2)
            prior_knowledge_score = round(random.uniform(*profile['prior_knowledge_range']), 2)
            study_frequency = random.choice(profile['study_frequencies'])
            attendance_rate = round(random.uniform(*profile['attendance_range']), 1)
            participation_score = round(random.uniform(*profile['participation_range']), 1)
            
            # Learning style affects interaction patterns
            learning_style = profile['learning_style']
            
            # Determine login pattern based on study frequency
            if study_frequency == 'daily':
                last_login_delta = random.randint(0, 3)
            elif study_frequency == 'weekly':
                last_login_delta = random.randint(0, 7)
            else:
                last_login_delta = random.randint(0, 14)
            
            # Calculate time spent based on study frequency and participation
            if study_frequency == 'daily':
                base_hours = random.randint(40, 80)
            elif study_frequency == 'weekly':
                base_hours = random.randint(20, 60)
            elif study_frequency == 'biweekly':
                base_hours = random.randint(15, 40)
            else:
                base_hours = random.randint(10, 30)
            
            # Adjust for participation level
            total_hours = int(base_hours * (0.8 + (participation_score/10) * 0.4))
            
            # Create student with profile data
            student = Student.objects.create(
                student_id=student_id,
                major=major,
                academic_level=academic_level,
                gpa=gpa,
                prior_knowledge_score=prior_knowledge_score,
                study_frequency=study_frequency,
                attendance_rate=attendance_rate,
                participation_score=participation_score,
                last_login_date=timezone.now() - datetime.timedelta(days=last_login_delta),
                total_time_spent=datetime.timedelta(hours=total_hours),
                average_time_per_session=datetime.timedelta(minutes=random.randint(30, 120)),
                # Add learning style metadata as appropriate for your model
                # learning_style=learning_style,
                # strengths=','.join(profile['strengths']),
                # weaknesses=','.join(profile['weaknesses'])
            )
            
            # Enroll in the course
            student.courses.add(course)
            students.append(student)
        
        return students

    def create_learning_progression_data(self, students, questions, resources, topic_prereqs):
        """Create realistic learning progression data for students"""
        interactions = []
        
        # Group questions by topic
        topic_questions = {}
        for q in questions:
            if q.topic not in topic_questions:
                topic_questions[q.topic] = []
            topic_questions[q.topic].append(q)
        
        # For each student, create realistic learning progression
        for student in students:
            # Get student's learning profile data
            gpa = student.gpa 
            prior_knowledge = student.prior_knowledge_score
            study_frequency = student.study_frequency
            
            # Set base mastery based on student profile
            base_mastery = min(0.7, gpa / 5.0 + prior_knowledge / 2)
            
            # Determine topic order based on prerequisites
            all_topics = set(topic_prereqs.keys())
            completed_topics = set()
            topic_order = []
            
            # Build topic order respecting prerequisites
            while len(completed_topics) < len(all_topics):
                for topic in all_topics:
                    if topic in completed_topics:
                        continue
                    
                    # Check if all prerequisites are completed
                    prereqs = set(topic_prereqs.get(topic, []))
                    if prereqs.issubset(completed_topics):
                        topic_order.append(topic)
                        completed_topics.add(topic)
                
                # Break if we can't add any more topics (should not happen with valid prereqs)
                if len(topic_order) == len(completed_topics):
                    break
            
            # Process topics in order
            topic_mastery = {}
            for topic_name in topic_order:
                # Find topic object
                topic = next((t for t in topic_questions.keys() if t.name == topic_name), None)
                if not topic or topic not in topic_questions or not topic_questions[topic]:
                    continue
                
                # Apply learning curve based on prerequisites
                prereq_boost = 0
                for prereq in topic_prereqs.get(topic_name, []):
                    if prereq in topic_mastery:
                        prereq_boost += topic_mastery[prereq] * 0.2
                
                # Calculate initial topic mastery
                initial_mastery = max(0.1, min(0.4, base_mastery - 0.3 + prereq_boost))
                
                # Determine number of interactions based on student profile and topic difficulty
                topic_difficulty = getattr(topic, 'difficulty_level', 3)
                if not isinstance(topic_difficulty, (int, float)):
                    topic_difficulty = 3
                
                if study_frequency == 'daily':
                    base_interactions = random.randint(5, 8)
                elif study_frequency == 'weekly':
                    base_interactions = random.randint(3, 6)
                else:
                    base_interactions = random.randint(2, 4)
                
                # More interactions for difficult topics
                num_interactions = base_interactions + math.floor(topic_difficulty / 2)
                
                # Create learning progression with realistic mastery curve
                current_mastery = initial_mastery
                topic_qs = topic_questions[topic]
                
                for attempt in range(1, num_interactions + 1):
                    # Pick a question
                    question = random.choice(topic_qs)
                    
                    # Learning curve follows sigmoid function
                    progress_factor = attempt / num_interactions
                    sigmoid = 1 / (1 + math.exp(-10 * (progress_factor - 0.5)))
                    mastery_gain = (1 - initial_mastery) * sigmoid
                    current_mastery = initial_mastery + mastery_gain
                    
                    # Add random noise to make it realistic
                    current_mastery = min(1.0, max(0.1, current_mastery + random.uniform(-0.1, 0.1)))
                    
                    # Determine correctness based on current mastery
                    correct = random.random() < current_mastery
                    
                    # Score reflects mastery level with some variability
                    if correct:
                        score = int(70 + (current_mastery * 30) + random.randint(-5, 5))
                    else:
                        score = int(30 + (current_mastery * 40) + random.randint(-10, 10))
                    score = max(0, min(100, score))
                    
                    # Time taken decreases with mastery
                    base_minutes = 15 - (current_mastery * 10) + (topic_difficulty * 2)
                    time_variance = random.uniform(-0.2, 0.2) * base_minutes
                    time_taken = datetime.timedelta(minutes=max(1, base_minutes + time_variance))
                    
                    # More likely to view resources on early attempts or when struggling
                    resource_viewed = random.random() < (0.8 - current_mastery + 0.2)
                    
                    # Create timestamp - earlier attempts happen earlier
                    days_ago = 60 - (attempt * 5) + random.randint(-3, 3)
                    timestamp = timezone.now() - datetime.timedelta(days=max(0, days_ago))
                    
                    # Create interaction
                    interaction = StudentInteraction.objects.create(
                        student=student,
                        question=question,
                        response=f"Student response for {topic.name} question {attempt}",
                        correct=correct,
                        score=score,
                        time_taken=time_taken,
                        timestamp=timestamp,
                        attempt_number=attempt,
                        resource_viewed_before=resource_viewed
                    )
                    interactions.append(interaction)
                
                # Store final mastery for this topic (affects prerequisites)
                topic_mastery[topic_name] = current_mastery
        
        return interactions