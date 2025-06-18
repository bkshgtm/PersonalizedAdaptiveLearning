import json
import os
from datetime import timedelta
from django.core.management.base import BaseCommand
from django.db import transaction
from django.conf import settings
from core.models import Resource, Topic, Course


class Command(BaseCommand):
    help = 'Load learning resources from java_resources.json'

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
        self.stdout.write(self.style.SUCCESS('Starting to load learning resources...'))
        
        # Path to the JSON file
        json_file_path = os.path.join(settings.BASE_DIR, 'static', 'data', 'java_resources.json')
        
        if not os.path.exists(json_file_path):
            self.stdout.write(self.style.ERROR(f'File not found: {json_file_path}'))
            return
        
        with transaction.atomic():
            # Load JSON data
            resources_data = self.load_json_data(json_file_path)
            if not resources_data:
                return
            
            # Get the CS206 course to validate topics belong to it
            try:
                course = Course.objects.get(course_id='CS206')
                self.stdout.write(f'Found course: {course.title}')
            except Course.DoesNotExist:
                self.stdout.write(self.style.ERROR('CS206 course not found. Please run previous setup scripts first.'))
                return
            
            # Create topic ID to Topic object mapping for efficient lookup
            topic_mapping = self.create_topic_mapping(course)
            
            # Process resources
            resources_created, resources_updated, resources_skipped = self.process_resources(
                resources_data.get('resources', []), topic_mapping
            )
        
        self.stdout.write(self.style.SUCCESS(
            f'Resource loading complete! '
            f'Created: {resources_created}, Updated: {resources_updated}, Skipped: {resources_skipped}'
        ))

    def load_json_data(self, json_file_path):
        """Load and validate JSON data from file"""
        try:
            with open(json_file_path, 'r') as file:
                data = json.load(file)
            
            resources_count = len(data.get('resources', []))
            self.stdout.write(f'Loaded JSON with {resources_count} resources')
            return data
            
        except json.JSONDecodeError as e:
            self.stdout.write(self.style.ERROR(f'Error parsing JSON file: {e}'))
            return None
        except FileNotFoundError:
            self.stdout.write(self.style.ERROR(f'File not found: {json_file_path}'))
            return None

    def create_topic_mapping(self, course):
        """Create mapping from topic IDs in JSON to actual Topic objects"""
        # Get all topics for the course
        topics = Topic.objects.filter(course=course)
        
        # Create mapping based on the order topics were created (assuming JSON IDs 1-20 map to our topics)
        # Since we can't rely on primary keys, we'll use the JSON structure to map IDs to topic names
        topic_mapping = {}
        
        # Load the KSG JSON to get the mapping of IDs to topic names
        ksg_file_path = os.path.join(settings.BASE_DIR, 'static', 'data', 'java_ksg.json')
        try:
            with open(ksg_file_path, 'r') as file:
                ksg_data = json.load(file)
            
            # Create ID to name mapping from KSG data
            id_to_name = {}
            for node in ksg_data.get('nodes', []):
                id_to_name[node['id']] = node['name']
            
            # Now map JSON IDs to actual Topic objects using names
            for json_id, topic_name in id_to_name.items():
                try:
                    topic = topics.get(name=topic_name)
                    topic_mapping[json_id] = topic
                except Topic.DoesNotExist:
                    self.stdout.write(self.style.WARNING(f'Topic not found: {topic_name} (ID: {json_id})'))
            
            self.stdout.write(f'Created topic mapping for {len(topic_mapping)} topics')
            return topic_mapping
            
        except (FileNotFoundError, json.JSONDecodeError) as e:
            self.stdout.write(self.style.ERROR(f'Error loading KSG data for topic mapping: {e}'))
            return {}

    def process_resources(self, resources_data, topic_mapping):
        """Process each resource from the JSON data"""
        resources_created = 0
        resources_updated = 0
        resources_skipped = 0
        
        for resource_data in resources_data:
            try:
                # Validate required fields
                title = resource_data.get('title')
                url = resource_data.get('url')
                resource_type = resource_data.get('resource_type')
                topic_ids = resource_data.get('topics', [])
                
                if not all([title, url, resource_type]):
                    self.stdout.write(self.style.WARNING(
                        f'Skipping resource with missing required fields: {resource_data.get("id", "unknown")}'
                    ))
                    resources_skipped += 1
                    continue
                
                # Validate topic IDs exist in our mapping
                valid_topics = []
                for topic_id in topic_ids:
                    if topic_id in topic_mapping:
                        valid_topics.append(topic_mapping[topic_id])
                    else:
                        self.stdout.write(self.style.ERROR(
                            f'Invalid topic ID {topic_id} for resource: {title}'
                        ))
                        # This is a critical error - raise exception to stop processing
                        raise ValueError(f'Topic ID {topic_id} does not map to any existing Topic')
                
                if not valid_topics:
                    self.stdout.write(self.style.WARNING(f'No valid topics found for resource: {title}'))
                    resources_skipped += 1
                    continue
                
                # Check if resource already exists (idempotency)
                existing_resource = Resource.objects.filter(title=title, url=url).first()
                
                if existing_resource:
                    # Update existing resource
                    updated = self.update_resource(existing_resource, resource_data, valid_topics)
                    if updated:
                        resources_updated += 1
                        action = '[DRY RUN] Would update' if self.dry_run else 'Updated'
                        self.stdout.write(f'{action} resource: {title}')
                    else:
                        resources_skipped += 1
                        self.stdout.write(f'Resource unchanged: {title}')
                else:
                    # Create new resource
                    if not self.dry_run:
                        self.create_resource(resource_data, valid_topics)
                    resources_created += 1
                    action = '[DRY RUN] Would create' if self.dry_run else 'Created'
                    self.stdout.write(f'{action} resource: {title}')
                
            except Exception as e:
                self.stdout.write(self.style.ERROR(
                    f'Error processing resource {resource_data.get("id", "unknown")}: {e}'
                ))
                # Re-raise critical errors
                if isinstance(e, ValueError):
                    raise
                continue
        
        return resources_created, resources_updated, resources_skipped

    def create_resource(self, resource_data, topics):
        """Create a new resource with the given data and topics"""
        # Parse estimated time if provided
        estimated_time = self.parse_estimated_time(resource_data.get('estimated_time'))
        
        # Validate difficulty level
        difficulty = resource_data.get('difficulty', 'beginner')
        valid_difficulties = ['beginner', 'intermediate', 'advanced']
        if difficulty not in valid_difficulties:
            difficulty = 'beginner'
        
        # Validate resource type
        resource_type = resource_data.get('resource_type')
        valid_types = ['video', 'document', 'exercise', 'tutorial', 'quiz', 'example', 
                      'reference', 'documentation', 'course', 'book', 'practice', 'visual', 'other']
        if resource_type not in valid_types:
            self.stdout.write(self.style.WARNING(
                f'Unknown resource type "{resource_type}" for {resource_data.get("title")}. Using "other".'
            ))
            resource_type = 'other'
        
        # Create the resource
        resource = Resource.objects.create(
            title=resource_data.get('title'),
            description=resource_data.get('description', ''),
            url=resource_data.get('url'),
            resource_type=resource_type,
            difficulty=difficulty,
            estimated_time=estimated_time
        )
        
        # Add topics to the resource (ManyToMany relationship)
        resource.topics.set(topics)
        
        return resource

    def update_resource(self, resource, resource_data, topics):
        """Update an existing resource if any fields have changed"""
        updated = False
        
        # Check and update fields
        new_description = resource_data.get('description', '')
        if resource.description != new_description:
            if not self.dry_run:
                resource.description = new_description
            updated = True
        
        new_difficulty = resource_data.get('difficulty', 'beginner')
        if new_difficulty in ['beginner', 'intermediate', 'advanced'] and resource.difficulty != new_difficulty:
            if not self.dry_run:
                resource.difficulty = new_difficulty
            updated = True
        
        new_estimated_time = self.parse_estimated_time(resource_data.get('estimated_time'))
        if resource.estimated_time != new_estimated_time:
            if not self.dry_run:
                resource.estimated_time = new_estimated_time
            updated = True
        
        # Check if topics have changed
        current_topics = set(resource.topics.all())
        new_topics = set(topics)
        if current_topics != new_topics:
            if not self.dry_run:
                resource.topics.set(topics)
            updated = True
        
        if updated and not self.dry_run:
            resource.save()
        
        return updated

    def parse_estimated_time(self, time_string):
        """Parse estimated time string into timedelta object"""
        if not time_string:
            return None
        
        try:
            # Handle common formats like "25 minutes", "1 hour", "1.5 hours"
            time_string = time_string.lower().strip()
            
            if 'minute' in time_string:
                # Extract number of minutes
                minutes = float(time_string.split()[0])
                return timedelta(minutes=minutes)
            elif 'hour' in time_string:
                # Extract number of hours
                hours = float(time_string.split()[0])
                return timedelta(hours=hours)
            else:
                # Try to extract just the number and assume minutes
                try:
                    minutes = float(time_string.split()[0])
                    return timedelta(minutes=minutes)
                except (ValueError, IndexError):
                    return None
        
        except (ValueError, AttributeError):
            self.stdout.write(self.style.WARNING(f'Could not parse estimated time: {time_string}'))
            return None
