import json
import yaml
import os
import random
from datetime import datetime, timedelta
from django.core.management.base import BaseCommand
from django.db import transaction
from django.conf import settings
from django.utils import timezone
from core.models import Question, StudentInteraction, Student, Assessment, Topic, Course


class Command(BaseCommand):
    help = 'Load questions and student interactions from java_questions.yaml'

    def add_arguments(self, parser):
        parser.add_argument(
            '--dry-run',
            action='store_true',
            help='Show what would be done without making changes',
        )

    def handle(self, *args, **options):
        self.dry_run = options['dry_run']
        if self.dry_run:
            self.stdout.write(self.style.WARNING('DRY RUN MODE - No changes will be made'))
        
        self.stdout.write(self.style.SUCCESS('Starting to load questions and student interactions...'))
        
        # Load YAML data
        yaml_file_path = os.path.join(settings.BASE_DIR, 'static', 'data', 'java_questions.yaml')
        
        if not os.path.exists(yaml_file_path):
            self.stdout.write(self.style.ERROR(f'File not found: {yaml_file_path}'))
            return
        
        with transaction.atomic():
            # Load and validate data
            questions_data = self.load_yaml_data(yaml_file_path)
            if not questions_data:
                return
            
            # Get required objects
            course, students, assessments, topics = self.get_database_objects()
            if not all([course, students, assessments, topics]):
                return
            
            # Create topic name mapping
            topic_mapping = {topic.name: topic for topic in topics}
            
            # Process questions and interactions
            questions_created, interactions_created = self.process_questions_and_interactions(
                questions_data, students, assessments, topic_mapping
            )
        
        self.stdout.write(self.style.SUCCESS(
            f'Successfully processed! Questions: {questions_created}, Interactions: {interactions_created}'
        ))

    def load_yaml_data(self, yaml_file_path):
        """Load and validate YAML data"""
        try:
            with open(yaml_file_path, 'r') as file:
                data = yaml.safe_load(file)
            
            questions_count = len(data.get('questions', []))
            self.stdout.write(f'Loaded YAML with {questions_count} questions')
            return data
            
        except yaml.YAMLError as e:
            self.stdout.write(self.style.ERROR(f'Error parsing YAML file: {e}'))
            return None
        except FileNotFoundError:
            self.stdout.write(self.style.ERROR(f'File not found: {yaml_file_path}'))
            return None

    def get_database_objects(self):
        """Get all required database objects"""
        try:
            course = Course.objects.get(course_id='CS206')
            students = list(Student.objects.all())
            assessments = list(Assessment.objects.filter(course=course))
            topics = list(Topic.objects.filter(course=course))
            
            self.stdout.write(f'Found: {len(students)} students, {len(assessments)} assessments, {len(topics)} topics')
            
            if len(students) < 10:
                self.stdout.write(self.style.ERROR('Need at least 10 students. Run populate_users_courses first.'))
                return None, None, None, None
            
            if len(assessments) == 0:
                self.stdout.write(self.style.ERROR('No assessments found. Run load_assessments first.'))
                return None, None, None, None
                
            if len(topics) == 0:
                self.stdout.write(self.style.ERROR('No topics found. Run load_ksg first.'))
                return None, None, None, None
            
            return course, students, assessments, topics
            
        except Course.DoesNotExist:
            self.stdout.write(self.style.ERROR('CS206 course not found. Run populate_users_courses first.'))
            return None, None, None, None

    def process_questions_and_interactions(self, questions_data, students, assessments, topic_mapping):
        """Process questions and create realistic student interactions"""
        questions_created = 0
        interactions_created = 0
        
        # Create base timestamp for realistic progression
        base_date = timezone.now() - timedelta(days=120)  # Start 4 months ago
        
        # Track student learning progression for realism
        student_progress = {student.student_id: {
            'current_date': base_date + timedelta(days=random.randint(0, 7)),
            'skill_level': random.uniform(0.3, 0.7),  # Starting skill level
            'study_consistency': random.uniform(0.6, 0.95),  # How consistent they are
            'topic_mastery': {},  # Track mastery per topic
        } for student in students}
        
        for question_data in questions_data.get('questions', []):
            try:
                # Create Question object
                question = self.create_question(question_data, assessments, topic_mapping)
                if question:
                    questions_created += 1
                    
                    # Create student interactions for this question
                    interactions_count = self.create_student_interactions(
                        question, question_data, students, student_progress
                    )
                    interactions_created += interactions_count
                    
            except Exception as e:
                self.stdout.write(self.style.ERROR(f'Error processing question {question_data.get("question_id", "unknown")}: {e}'))
                continue
        
        return questions_created, interactions_created

    def create_question(self, question_data, assessments, topic_mapping):
        """Create a Question object from YAML data"""
        question_id = question_data.get('question_id')
        text = question_data.get('text', '')
        question_type = question_data.get('question_type', 'short_answer')
        topic_name = question_data.get('topic')
        
        # Validate required fields
        if not all([question_id, text, topic_name]):
            self.stdout.write(self.style.WARNING(f'Skipping question with missing fields: {question_id}'))
            return None
        
        # Check if question already exists
        if not self.dry_run and Question.objects.filter(question_id=question_id).exists():
            self.stdout.write(f'Question {question_id} already exists, skipping...')
            return Question.objects.get(question_id=question_id)
        
        # Get topic
        topic = topic_mapping.get(topic_name)
        if not topic:
            self.stdout.write(self.style.WARNING(f'Topic not found: {topic_name} for question {question_id}'))
            return None
        
        # Assign to a random assessment (realistic distribution)
        assessment = self.assign_question_to_assessment(assessments, topic_name, question_type)
        
        if self.dry_run:
            self.stdout.write(f'[DRY RUN] Would create question: {question_id} -> {topic_name}')
            return None
        
        # Create question
        question = Question.objects.create(
            question_id=question_id,
            assessment=assessment,
            text=text,
            question_type=question_type,
            topic=topic
        )
        
        self.stdout.write(f'Created question: {question_id} -> {topic_name} ({assessment.title})')
        return question

    def assign_question_to_assessment(self, assessments, topic_name, question_type):
        """Intelligently assign question to appropriate assessment"""
        # Filter assessments by type preference
        if question_type in ['mcq', 'true_false', 'fill_blank']:
            preferred = [a for a in assessments if a.assessment_type in ['quiz', 'exam']]
        elif question_type in ['coding', 'code_analysis']:
            preferred = [a for a in assessments if a.assessment_type in ['assignment', 'project']]
        else:
            preferred = assessments
        
        # Use preferred assessments if available, otherwise any assessment
        candidates = preferred if preferred else assessments
        
        # Add some topic-based logic for more realism
        topic_keywords = topic_name.lower().split()
        for assessment in candidates:
            assessment_keywords = assessment.title.lower().split()
            if any(keyword in assessment_keywords for keyword in topic_keywords):
                return assessment
        
        # Fallback to random selection
        return random.choice(candidates)

    def create_student_interactions(self, question, question_data, students, student_progress):
        """Create realistic student interactions for a question"""
        if self.dry_run:
            return 10  # Simulate creating 10 interactions
        
        sample_answers = question_data.get('sample_student_answers', [])
        if len(sample_answers) < 10:
            self.stdout.write(self.style.WARNING(f'Question {question.question_id} has only {len(sample_answers)} sample answers'))
            return 0
        
        interactions_created = 0
        topic_name = question.topic.name
        
        # Shuffle students for random assignment
        available_students = students.copy()
        random.shuffle(available_students)
        
        for i, answer_data in enumerate(sample_answers[:10]):  # Use first 10 answers
            student = available_students[i]
            progress = student_progress[student.student_id]
            
            # Create realistic interaction
            interaction = self.create_realistic_interaction(
                student, question, answer_data, progress, topic_name
            )
            
            if interaction:
                interactions_created += 1
                # Update student progress based on this interaction
                self.update_student_progress(progress, topic_name, interaction.correct, interaction.score)
        
        return interactions_created

    def create_realistic_interaction(self, student, question, answer_data, progress, topic_name):
        """Create a single realistic student interaction"""
        # Extract answer data
        response_text = answer_data.get('text', '')
        is_correct = answer_data.get('correct', False)
        base_score = answer_data.get('score', 0.0)
        
        # Add realism to timing
        interaction_time = self.generate_realistic_timestamp(progress, topic_name)
        time_taken = self.generate_realistic_time_taken(question.question_type, is_correct, progress['skill_level'])
        
        # Adjust score based on student's skill level and consistency
        adjusted_score = self.adjust_score_for_student(base_score, progress['skill_level'], is_correct)
        
        # Determine attempt number (some students retry)
        attempt_number = self.determine_attempt_number(student, question, progress['study_consistency'])
        
        # Check for existing interaction to avoid duplicates
        existing = StudentInteraction.objects.filter(
            student=student,
            question=question,
            attempt_number=attempt_number
        ).first()
        
        if existing:
            return existing
        
        # Create interaction
        interaction = StudentInteraction.objects.create(
            student=student,
            question=question,
            response=response_text,
            correct=is_correct,
            score=adjusted_score,
            time_taken=time_taken,
            timestamp=interaction_time,
            attempt_number=attempt_number,
            resource_viewed_before=self.did_student_view_resources(progress['skill_level'], is_correct)
        )
        
        return interaction

    def generate_realistic_timestamp(self, progress, topic_name):
        """Generate realistic timestamp based on student progress and topic"""
        # Students progress through topics over time
        current_date = progress['current_date']
        
        # Add some randomness but maintain progression
        days_variation = random.randint(-2, 5)  # Can review old topics
        hours_variation = random.randint(8, 22)  # Study during reasonable hours
        
        timestamp = current_date + timedelta(days=days_variation, hours=hours_variation)
        
        # Update student's current date for next question
        progress['current_date'] = max(progress['current_date'], timestamp + timedelta(hours=random.randint(1, 48)))
        
        return timestamp

    def generate_realistic_time_taken(self, question_type, is_correct, skill_level):
        """Generate realistic time taken based on question type and student ability"""
        # Base times by question type (in seconds)
        base_times = {
            'mcq': (30, 120),
            'true_false': (15, 60),
            'fill_blank': (45, 180),
            'short_answer': (120, 600),
            'coding': (300, 1800),
            'code_analysis': (180, 900)
        }
        
        min_time, max_time = base_times.get(question_type, (60, 300))
        
        # Adjust based on skill level
        skill_multiplier = 2.0 - skill_level  # Higher skill = less time
        
        # Adjust based on correctness (wrong answers often take longer due to confusion)
        correctness_multiplier = 0.8 if is_correct else 1.3
        
        # Calculate final time
        adjusted_min = min_time * skill_multiplier * correctness_multiplier
        adjusted_max = max_time * skill_multiplier * correctness_multiplier
        
        # Add randomness
        time_seconds = random.uniform(adjusted_min, adjusted_max)
        
        return timedelta(seconds=int(time_seconds))

    def adjust_score_for_student(self, base_score, skill_level, is_correct):
        """Adjust score based on student's skill level"""
        if not is_correct:
            return base_score  # Keep original score for incorrect answers
        
        # For correct answers, adjust based on skill level
        # Higher skill students tend to get higher scores on correct answers
        skill_bonus = (skill_level - 0.5) * 0.2  # -0.1 to +0.1 adjustment
        adjusted_score = min(1.0, max(0.0, base_score + skill_bonus))
        
        return round(adjusted_score, 2)

    def determine_attempt_number(self, student, question, study_consistency):
        """Determine if this is a retry attempt"""
        # Check if student already has attempts for this question
        existing_attempts = StudentInteraction.objects.filter(
            student=student,
            question=question
        ).count()
        
        if existing_attempts == 0:
            return 1
        
        # Some students retry based on their consistency
        retry_probability = study_consistency * 0.3  # Max 30% chance of retry
        if random.random() < retry_probability:
            return existing_attempts + 1
        
        return existing_attempts + 1

    def did_student_view_resources(self, skill_level, is_correct):
        """Determine if student viewed resources before answering"""
        # Lower skill students more likely to view resources
        # Students who got it wrong more likely to have viewed resources (but still got it wrong)
        base_probability = 0.7 - (skill_level * 0.4)  # 0.3 to 0.7 range
        
        if not is_correct:
            base_probability += 0.2  # More likely to have viewed resources if still wrong
        
        return random.random() < base_probability

    def update_student_progress(self, progress, topic_name, is_correct, score):
        """Update student's learning progress"""
        # Update topic mastery
        if topic_name not in progress['topic_mastery']:
            progress['topic_mastery'][topic_name] = []
        
        progress['topic_mastery'][topic_name].append(score)
        
        # Update overall skill level based on recent performance
        if is_correct:
            progress['skill_level'] = min(1.0, progress['skill_level'] + 0.01)
        else:
            progress['skill_level'] = max(0.1, progress['skill_level'] - 0.005)
