import json
import datetime
import re
from django.core.management.base import BaseCommand
from core.models import Resource, Topic

class Command(BaseCommand):
    help = 'Load resources from JSON file into the database'

    def add_arguments(self, parser):
        parser.add_argument('--file', type=str, default='static/data/resources json.txt', 
                           help='Path to resources JSON file')
        parser.add_argument('--clear', action='store_true', 
                           help='Clear existing resources before loading')
        parser.add_argument('--debug', action='store_true',
                           help='Print detailed debug information')

    def handle(self, *args, **options):
        file_path = options['file']
        clear_existing = options['clear']
        debug_mode = options.get('debug', False)
        
        # First, make sure we have at least one course
        from core.models import Course
        course = Course.objects.first()
        if not course:
            # Create a default course if none exists
            course = Course.objects.create(
                course_id='CS206',
                title='Java Programming',
                description='Introduction to Java Programming'
            )
            self.stdout.write(self.style.SUCCESS(f'Created default course: {course.title}'))
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            self.stdout.write(self.style.ERROR(f'Error loading file: {str(e)}'))
            return
        
        # Load topics from java_topics.json to ensure we have all the topics
        topics_path = 'static/data/java_topics.json'
        try:
            with open(topics_path, 'r') as f:
                topics_data = json.load(f)
                
            # Create topics if they don't exist
            for topic_data in topics_data:
                topic_id = topic_data.get('id')
                topic_name = topic_data.get('name')
                
                if topic_id and topic_name:
                    Topic.objects.get_or_create(
                        id=topic_id,
                        defaults={
                            'name': topic_name,
                            'description': f'Topic about {topic_name}',
                            'course': course  # Associate with the course
                        }
                    )
            
            self.stdout.write(self.style.SUCCESS(f'Ensured all topics from {topics_path} exist'))
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'Error loading topics: {str(e)}'))
        
        if clear_existing:
            Resource.objects.all().delete()
            self.stdout.write('Cleared existing resources')
        
        resources_created = 0
        resources_updated = 0
        errors = []
        
        for resource_data in data['resources']:
            try:
                resource_id = resource_data.pop('id')
                topics_ids = resource_data.pop('topics')
                
                # Handle estimated_time field
                est_time = resource_data.get('estimated_time')
                if est_time in ['reference', 'varies'] or not est_time:
                    resource_data['estimated_time'] = datetime.timedelta(minutes=60)
                else:
                    # If it's not a valid time format, set to None
                    if isinstance(est_time, str):
                        if re.search(r'\d+\s*(hours|hour|minutes|minute|mins|min)', est_time, re.IGNORECASE):
                            # Valid time format, convert it
                            if 'hour' in est_time.lower() and 'minute' in est_time.lower():
                                # Format: "X hours Y minutes"
                                hours_match = re.search(r'(\d+)\s*(hours|hour)', est_time, re.IGNORECASE)
                                minutes_match = re.search(r'(\d+)\s*(minutes|minute|mins|min)', est_time, re.IGNORECASE)
                                
                                hours = int(hours_match.group(1)) if hours_match else 0
                                minutes = int(minutes_match.group(1)) if minutes_match else 0
                                
                                resource_data['estimated_time'] = datetime.timedelta(hours=hours, minutes=minutes)
                            elif 'hour' in est_time.lower():
                                # Format: "X hours"
                                hours_match = re.search(r'(\d+)\s*(hours|hour)', est_time, re.IGNORECASE)
                                hours = int(hours_match.group(1)) if hours_match else 0
                                resource_data['estimated_time'] = datetime.timedelta(hours=hours)
                            elif 'minute' in est_time.lower() or 'min' in est_time.lower():
                                # Format: "X minutes"
                                minutes_match = re.search(r'(\d+)\s*(minutes|minute|mins|min)', est_time, re.IGNORECASE)
                                minutes = int(minutes_match.group(1)) if minutes_match else 0
                                resource_data['estimated_time'] = datetime.timedelta(minutes=minutes)
                        else:
                            # Not a valid time format, set to default
                            resource_data['estimated_time'] = datetime.timedelta(minutes=60)
            
                # Fix resource_type if it's too long
                if 'resource_type' in resource_data and len(resource_data['resource_type']) > 20:
                    resource_type = resource_data['resource_type'].lower()
                    if 'video' in resource_type:
                        resource_data['resource_type'] = 'video'
                    elif 'document' in resource_type:
                        resource_data['resource_type'] = 'document'
                    elif 'tutorial' in resource_type:
                        resource_data['resource_type'] = 'tutorial'
                    elif 'exercise' in resource_type or 'practice' in resource_type:
                        resource_data['resource_type'] = 'practice'
                    elif 'quiz' in resource_type or 'test' in resource_type:
                        resource_data['resource_type'] = 'quiz'
                    elif 'documentation' in resource_type:
                        resource_data['resource_type'] = 'documentation'
                    elif 'book' in resource_type:
                        resource_data['resource_type'] = 'book'
                    elif 'course' in resource_type:
                        resource_data['resource_type'] = 'course'
                    elif 'article' in resource_type:
                        resource_data['resource_type'] = 'article'
                    else:
                        resource_data['resource_type'] = 'document'
            
                # Fix difficulty if it's too long
                if 'difficulty' in resource_data and len(resource_data['difficulty']) > 20:
                    difficulty = resource_data['difficulty'].lower()
                    if 'beginner' in difficulty and 'advanced' in difficulty:
                        resource_data['difficulty'] = 'advanced'
                    elif 'beginner' in difficulty:
                        resource_data['difficulty'] = 'beginner'
                    elif 'intermediate' in difficulty:
                        resource_data['difficulty'] = 'intermediate'
                    elif 'advanced' in difficulty:
                        resource_data['difficulty'] = 'advanced'
                    else:
                        resource_data['difficulty'] = 'intermediate'
            
                # Create or update resource
                resource, created = Resource.objects.update_or_create(
                    id=resource_id,
                    defaults=resource_data
                )
                
                # Add topics - ensure they exist first
                valid_topics = []
                for topic_id in topics_ids:
                    topic = Topic.objects.filter(id=topic_id).first()
                    if topic:
                        valid_topics.append(topic)
                    else:
                        # Try to create the topic if it doesn't exist
                        try:
                            topic = Topic.objects.create(
                                id=topic_id,
                                name=f"Topic {topic_id}",
                                description=f"Auto-created topic {topic_id}",
                                course=course
                            )
                            valid_topics.append(topic)
                            self.stdout.write(self.style.SUCCESS(
                                f"Created missing topic ID {topic_id} for resource {resource_id}"
                            ))
                        except Exception as e:
                            self.stdout.write(self.style.WARNING(
                                f"Topic ID {topic_id} not found for resource {resource_id}"
                            ))
                
                if valid_topics:
                    resource.topics.set(valid_topics)
                else:
                    self.stdout.write(self.style.WARNING(
                        f"No valid topics found for resource {resource_id}"
                    ))
                
                if created:
                    resources_created += 1
                else:
                    resources_updated += 1
                
            except Exception as e:
                errors.append(f"Error processing resource {resource_data.get('id', 'unknown')}: {str(e)}")
                self.stdout.write(self.style.ERROR(f"Error processing resource: {str(e)}"))
        
        # Report results
        self.stdout.write(self.style.SUCCESS(
            f'Successfully loaded {resources_created} new resources and updated {resources_updated} existing resources'
        ))
        
        if errors:
            self.stdout.write(self.style.WARNING(f'Encountered {len(errors)} errors:'))
            for error in errors[:10]:  # Show first 10 errors
                self.stdout.write(f"  - {error}")
            if len(errors) > 10:
                self.stdout.write(f"  ... and {len(errors) - 10} more errors")
