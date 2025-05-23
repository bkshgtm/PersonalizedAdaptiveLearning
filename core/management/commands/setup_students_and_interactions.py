import logging
import random
import datetime
import yaml
import json
import os
from django.core.management.base import BaseCommand, CommandError
from django.utils import timezone
from django.db import transaction

from core.models import Student, Course, Question, StudentInteraction, Assessment, Topic

logger = logging.getLogger(__name__)

class Command(BaseCommand):
    help = 'Setup students and create interactions using Java questions YAML data'

    def add_arguments(self, parser):
        parser.add_argument('course_id', type=str, help='Course ID to enroll students in')
        parser.add_argument('--num_students', type=int, default=10, help='Number of students to create')
        parser.add_argument('--id_prefix', type=str, default='A00', help='Prefix for student IDs')
        parser.add_argument('--questions_file', type=str, default='/app/static/data/java_questions.yaml', 
                           help='Path to YAML file with Java questions')
        parser.add_argument('--topics_file', type=str, default='/app/static/data/java_topics.json', 
                           help='Path to JSON file with Java topics structure')
        parser.add_argument('--interactions_per_student', type=int, default=30, 
                           help='Target number of interactions to create per student (may be exceeded to use all sample answers)')

    def handle(self, *args, **options):
        course_id = options['course_id']
        num_students = options['num_students']
        id_prefix = options['id_prefix']
        questions_file = options['questions_file']
        topics_file = options['topics_file']
        interactions_per_student = options['interactions_per_student']

        try:
            # Load the course
            course = Course.objects.get(course_id=course_id)
            self.stdout.write(f"Found course: {course.title}")
            
            # Load Java questions
            questions_data = self.load_questions(questions_file)
            self.stdout.write(f"Loaded {len(questions_data)} questions from {questions_file}")
            
            # Load Java topics
            topics_data = self.load_topics(topics_file)
            self.stdout.write(f"Loaded topics structure from {topics_file}")
            
            # Delete all existing students and their interactions
            deleted_count, _ = Student.objects.all().delete()
            self.stdout.write(f"Deleted {deleted_count} existing students and their interactions")
            
            # Create new students
            students = self.create_students(course, num_students, id_prefix)
            
            # Create or update topics based on the topics file
            topics_map = self.create_topics(course, topics_data)
            
            # Create questions and assessments
            questions_map = self.create_questions(course, questions_data, topics_map)
            
            # Create interactions for each student
            total_interactions = self.create_interactions(
                students, questions_data, questions_map, interactions_per_student
            )
            
            self.stdout.write(self.style.SUCCESS(
                f"Successfully created {len(students)} students and {total_interactions} interactions"
            ))
            
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Error: {str(e)}"))
            logger.exception("Error in setup_students_and_interactions command")
    
    def load_questions(self, file_path):
        """Load questions from YAML file"""
        try:
            if not os.path.exists(file_path):
                self.stdout.write(self.style.ERROR(f"Questions file not found: {file_path}"))
                self.stdout.write(f"Current directory: {os.getcwd()}")
                self.stdout.write(f"Directory contents: {os.listdir(os.path.dirname(file_path) if os.path.dirname(file_path) else '.')}")
                return []
            
            with open(file_path, 'r') as file:
                data = yaml.safe_load(file)
                # If the data is a list, return it directly
                if isinstance(data, list):
                    return data
                # If it's a dict with a 'questions' key, return that
                return data.get('questions', [])
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Error loading questions file: {str(e)}"))
            return []
    
    def load_topics(self, file_path):
        """Load topics from JSON file"""
        try:
            if os.path.exists(file_path):
                with open(file_path, 'r') as file:
                    return json.load(file)
            else:
                self.stdout.write(self.style.WARNING(f"Topics file not found: {file_path}"))
                return {}
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Error loading topics file: {str(e)}"))
            return {}
    
    def create_students(self, course, num_students, id_prefix):
        """Create students with specific ID pattern and enroll them in the course"""
        # Use the exact choices from the Student model
        majors = ['Computer Science', 'Information Technology', 'Software Engineering', 
                  'Data Science', 'Cybersecurity', 'Digital Media', 'Game Development']
        
        # Use the exact choices from the Student model
        academic_levels = [choice[0] for choice in Student.ACADEMIC_LEVEL_CHOICES]
        study_frequencies = [choice[0] for choice in Student.STUDY_FREQUENCY_CHOICES]
        
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
        
        students = []
        
        for i in range(num_students):
            # Select a profile for this student
            profile = random.choice(student_profiles)
            
            # Generate student ID with format A00XXXXXX (where X is a digit)
            student_id = f"{id_prefix}{random.randint(100000, 999999)}"
            
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
                user=None,  # No user association for generated students
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
            students.append(student)
            
            self.stdout.write(f"Created student {student_id} ({profile['learning_style']} learner)")
        
        return students
    
    def create_topics(self, course, topics_data):
        """Get existing topics based on the topics file"""
        topics_map = {}
        
        # Load all existing topics for this course
        existing_topics = Topic.objects.filter(course=course)
        
        if not existing_topics.exists():
            self.stdout.write(self.style.WARNING(
                f"No topics found for course {course.course_id}. Questions will not be properly categorized."
            ))
            return topics_map
        
        # Map all existing topics by name
        for topic in existing_topics:
            topics_map[topic.name] = topic
        
        self.stdout.write(f"Loaded {len(topics_map)} existing topics for course {course.course_id}")
        
        return topics_map
    
    def create_questions(self, course, questions_data, topics_map):
        """Create questions and assessments from the questions data"""
        questions_map = {}
        assessments_map = {}
        
        # First, load all existing topics for this course to avoid creating duplicates
        all_topics = {topic.name: topic for topic in Topic.objects.filter(course=course)}
        topics_map.update(all_topics)
        
        for question_data in questions_data:
            question_id = question_data.get('question_id')
            if not question_id:
                continue
            
            topic_name = question_data.get('topic', 'General Java')
            
            # Get topic if it exists, or skip this question
            if topic_name in topics_map:
                topic = topics_map[topic_name]
            else:
                self.stdout.write(self.style.WARNING(
                    f"Topic '{topic_name}' not found for question {question_id}. Using default topic."
                ))
                # Try to get a default topic
                default_topics = Topic.objects.filter(course=course)
                if default_topics.exists():
                    topic = default_topics.first()
                    topics_map[topic_name] = topic
                else:
                    self.stdout.write(self.style.ERROR(
                        f"No topics found for course {course.course_id}. Skipping question {question_id}."
                    ))
                    continue
            
            # Create a unique assessment ID based on topic
            assessment_id = f"JAVA-{topic_name.replace(' ', '-').upper()}"
            
            # Get or create assessment
            if assessment_id not in assessments_map:
                assessment, created = Assessment.objects.get_or_create(
                    assessment_id=assessment_id,
                    course=course,
                    defaults={
                        'title': f"{topic_name} Assessment",
                        'assessment_type': 'quiz',
                        'date': timezone.now() - datetime.timedelta(days=random.randint(1, 30)),
                        'proctored': False
                    }
                )
                assessments_map[assessment_id] = assessment
            
            assessment = assessments_map[assessment_id]
            
            # Create or update question
            question, created = Question.objects.update_or_create(
                question_id=question_id,
                defaults={
                    'assessment': assessment,
                    'text': question_data.get('text', ''),
                    'question_type': question_data.get('question_type', 'mcq'),
                    'topic': topic
                }
            )
            
            questions_map[question_id] = question
            
            if created:
                self.stdout.write(f"Created question: {question_id}")
            else:
                self.stdout.write(f"Updated question: {question_id}")
        
        return questions_map
    
    def create_interactions(self, students, questions_data, questions_map, interactions_per_student):
        """Create interactions for students using the questions data"""
        total_interactions = 0
        
        # Track created interactions to avoid duplicates
        created_interactions = set()
        
        # Create a timeline for interactions (last 60 days)
        timeline = list(range(1, 61))
        random.shuffle(timeline)
        
        # Process each question
        for question_data in questions_data:
            question_id = question_data.get('question_id')
            if not question_id or question_id not in questions_map:
                self.stdout.write(self.style.WARNING(f"Skipping question {question_id} - not found in database"))
                continue
            
            question = questions_map[question_id]
            difficulty = question_data.get('difficulty', 3)
            
            # Get sample student answers from the YAML
            sample_answers = question_data.get('sample_student_answers', [])
            
            if not sample_answers:
                self.stdout.write(self.style.WARNING(f"Skipping question {question_id} - no sample answers"))
                continue
            
            # Distribute answers among students
            # If we have more answers than students, some students will get multiple answers
            # If we have more students than answers, some answers will be used multiple times
            student_assignments = []
            
            # First, assign each answer to a student
            for i, answer in enumerate(sample_answers):
                student_idx = i % len(students)
                student_assignments.append((students[student_idx], answer))
            
            # If we need more assignments to reach interactions_per_student * num_students,
            # add more random assignments
            total_desired = min(len(sample_answers) * len(students), interactions_per_student * len(students))
            while len(student_assignments) < total_desired:
                student = random.choice(students)
                answer = random.choice(sample_answers)
                student_assignments.append((student, answer))
            
            # Process each student-answer assignment
            for student, answer in student_assignments:
                # Determine student's ability level based on GPA and prior knowledge
                ability_level = (student.gpa / 4.0 * 0.7)
                if student.prior_knowledge_score is not None:
                    ability_level += (student.prior_knowledge_score * 0.3)
                
                # Get correctness from the answer
                correct = answer.get('correct', False)
                
                # Get score from the answer
                score = float(answer.get('score', 1.0 if correct else 0.0)) * 100
                
                # Calculate time taken based on difficulty and student ability
                base_time = difficulty * 2  # Base time in minutes
                ability_factor = 1.0 - (ability_level * 0.5)  # Higher ability = less time
                time_minutes = max(1, base_time * ability_factor * random.uniform(0.8, 1.2))
                
                # Create timestamp based on the timeline
                days_ago = timeline[total_interactions % len(timeline)]
                timestamp = timezone.now() - datetime.timedelta(days=days_ago)
                
                # Determine if resource was viewed (struggling students more likely to view resources)
                resource_viewed = random.random() < (1.0 - ability_level + 0.2)
                
                # Find the highest attempt number for this student-question pair
                existing_attempts = StudentInteraction.objects.filter(
                    student=student,
                    question=question
                ).values_list('attempt_number', flat=True)
                
                if existing_attempts:
                    # Use the next attempt number
                    attempt_number = max(existing_attempts) + 1
                else:
                    # First attempt
                    attempt_number = 1
                    
                # Check if we've already created this interaction in this run
                if (student.student_id, question.question_id, attempt_number) in created_interactions:
                    # Skip this one to avoid duplicates
                    continue
                    
                # Add to our tracking set
                created_interactions.add((student.student_id, question.question_id, attempt_number))
                
                # Create the interaction
                try:
                    interaction = StudentInteraction.objects.create(
                        student=student,
                        question=question,
                        response=answer.get('text', ''),
                        correct=correct,
                        score=score,
                        time_taken=datetime.timedelta(minutes=time_minutes),
                        timestamp=timestamp,
                        attempt_number=attempt_number,
                        resource_viewed_before=resource_viewed
                    )
                    
                    total_interactions += 1
                    
                    if total_interactions % 100 == 0:
                        self.stdout.write(f"Created {total_interactions} interactions...")
                except Exception as e:
                    self.stdout.write(self.style.WARNING(
                        f"Skipping duplicate interaction: {student.student_id}, {question.question_id}, {attempt_number}"
                    ))
        
        return total_interactions
