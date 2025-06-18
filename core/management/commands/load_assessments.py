import json
import os
from django.core.management.base import BaseCommand
from django.db import transaction
from django.utils.dateparse import parse_datetime
from django.conf import settings
from core.models import Assessment, Course


class Command(BaseCommand):
    help = 'Load assessment data from java_assessments.json'

    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS('Starting to load assessments...'))
        
        # Path to the JSON file
        json_file_path = os.path.join(settings.BASE_DIR, 'static', 'data', 'java_assessments.json')
        
        if not os.path.exists(json_file_path):
            self.stdout.write(self.style.ERROR(f'File not found: {json_file_path}'))
            return
        
        with transaction.atomic():
            # Get the CS206 course using name-based lookup
            try:
                course = Course.objects.get(course_id='CS206')
                self.stdout.write(f'Found course: {course.title}')
            except Course.DoesNotExist:
                self.stdout.write(self.style.ERROR('CS206 course not found. Please run populate_users_courses first.'))
                return
            
            # Load and process assessments
            assessments_created = self.load_assessments_from_json(json_file_path, course)
        
        self.stdout.write(self.style.SUCCESS(f'Successfully loaded {assessments_created} assessments!'))

    def load_assessments_from_json(self, json_file_path, course):
        """Load assessments from JSON file"""
        try:
            with open(json_file_path, 'r') as file:
                assessments_data = json.load(file)
        except json.JSONDecodeError as e:
            self.stdout.write(self.style.ERROR(f'Error parsing JSON file: {e}'))
            return 0
        except FileNotFoundError:
            self.stdout.write(self.style.ERROR(f'File not found: {json_file_path}'))
            return 0
        
        assessments_created = 0
        
        for assessment_data in assessments_data:
            try:
                # Check if assessment already exists (prevent duplicates)
                assessment_id = assessment_data.get('assessment_id')
                title = assessment_data.get('title')
                
                if Assessment.objects.filter(assessment_id=assessment_id).exists():
                    self.stdout.write(f'Assessment {assessment_id} already exists, skipping...')
                    continue
                
                # Parse the date
                date_str = assessment_data.get('date')
                date = parse_datetime(date_str) if date_str else None
                
                if not date:
                    self.stdout.write(self.style.WARNING(f'Invalid date for assessment {assessment_id}: {date_str}'))
                    continue
                
                # Create the assessment
                assessment = Assessment.objects.create(
                    assessment_id=assessment_id,
                    title=title,
                    assessment_type=assessment_data.get('assessment_type', 'quiz'),
                    course=course,  # Link to CS206 course using lookup, not hardcoded ID
                    date=date,
                    proctored=assessment_data.get('proctored', False)
                )
                
                assessments_created += 1
                self.stdout.write(f'Created assessment: {assessment_id} - {title}')
                
            except Exception as e:
                self.stdout.write(self.style.ERROR(f'Error creating assessment {assessment_data.get("assessment_id", "unknown")}: {e}'))
                continue
        
        return assessments_created
