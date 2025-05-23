import os
import django
import json
import yaml
from django.db import transaction

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'pal_project.settings')
django.setup()

from django.contrib.auth.models import User
from core.models import Course, Topic, Resource, Assessment, Question
from knowledge_graph.models import KnowledgeGraph, GraphEdge

def clean_field(value, max_length, default=None):
    """Clean and truncate a string field to fit within max_length"""
    if not value or not isinstance(value, str):
        return default
    
    if len(value) <= max_length:
        return value
    
    # Truncate the value
    return value[:max_length]

def load_required_data():
    """
    Load only the required data for the PAL system:
    1. Admin user
    2. Course
    3. Topics from java_topics.json
    4. Resources from resources json.txt
    5. Knowledge graph from sample_ksg.json
    6. Students and interactions
    """
    print("Loading required data for PAL system...")
    
    try:
        with transaction.atomic():
            # 1. Create admin user if it doesn't exist
            if not User.objects.filter(username='admin').exists():
                User.objects.create_superuser(
                    username='admin',
                    email='admin@example.com',
                    password='adminpassword'
                )
                print("Created admin user")
            else:
                print("Admin user already exists")
            
            admin_user = User.objects.get(username='admin')
            
            # 2. Create Java course
            course, created = Course.objects.get_or_create(
                course_id='CS206',
                defaults={
                    'title': 'Introduction to Java Programming',
                    'description': 'A comprehensive introduction to Java programming language and object-oriented concepts.'
                }
            )
            
            if created:
                print(f"Created course: {course.title}")
            else:
                print(f"Using existing course: {course.title}")
            
            # 3. Clear existing topics first
            Topic.objects.all().delete()
            print("Cleared existing topics")
            
            # Load topics from java_topics.json
            topics_path = 'static/data/java_topics.json'
            if os.path.exists(topics_path):
                with open(topics_path, 'r') as f:
                    topics_data = json.load(f)
                
                # Create topics
                for topic_data in topics_data:
                    topic_id = topic_data.get('id')
                    topic_name = topic_data.get('name')
                    topic_description = topic_data.get('description', f'Topic about {topic_name}')
                    
                    if topic_id and topic_name:
                        try:
                            topic = Topic.objects.create(
                                id=topic_id,
                                name=topic_name,
                                description=topic_description,
                                course=course
                            )
                            print(f"Created topic: {topic.name} (ID: {topic.id})")
                        except Exception as e:
                            print(f"Error creating topic {topic_name}: {str(e)}")
                            continue
                
                print(f"Created/updated {Topic.objects.count()} topics")
            else:
                print(f"Topics file not found: {topics_path}")
            
            # 4. Load resources from resources json.txt
            resources_path = 'static/data/resources json.txt'
            if os.path.exists(resources_path):
                with open(resources_path, 'r') as f:
                    resources_data = json.load(f)
                
                # Create resources
                for resource_data in resources_data.get('resources', []):
                    # Clean fields that have length constraints
                    if 'resource_type' in resource_data:
                        resource_data['resource_type'] = clean_field(resource_data['resource_type'], 50, 'document')
                    
                    if 'difficulty' in resource_data:
                        resource_data['difficulty'] = clean_field(resource_data['difficulty'], 20, 'intermediate')
                    
                    resource_id = resource_data.get('id')
                    
                    if not resource_id:
                        continue
                    
                    # Extract topic IDs
                    topic_ids = resource_data.pop('topics', [])
                    
                    # Create or update resource
                    resource, created = Resource.objects.update_or_create(
                        id=resource_id,
                        defaults=resource_data
                    )
                    
                    # Add topics
                    topics = Topic.objects.filter(id__in=topic_ids)
                    if topics.exists():
                        resource.topics.set(topics)
                    
                    if created:
                        print(f"Created resource: {resource.title} (ID: {resource.id})")
                
                print(f"Created/updated {Resource.objects.count()} resources")
            else:
                print(f"Resources file not found: {resources_path}")
            
            # 5. Load knowledge graph from sample_ksg.json
            ksg_path = 'static/data/sample_ksg.json'
            if os.path.exists(ksg_path):
                with open(ksg_path, 'r') as f:
                    ksg_data = json.load(f)
                
                # Create knowledge graph
                graph, created = KnowledgeGraph.objects.update_or_create(
                    name=ksg_data.get('name', 'Java Knowledge Graph'),
                    version=ksg_data.get('version', '1.0'),
                    defaults={
                        'description': ksg_data.get('description', 'Java Knowledge Structure Graph'),
                        'data': ksg_data,
                        'is_active': True,
                        'created_by': admin_user
                    }
                )
                
                if created:
                    print(f"Created knowledge graph: {graph.name}")
                else:
                    print(f"Updated knowledge graph: {graph.name}")
                
                # Create edges
                topic_map = {topic.id: topic for topic in Topic.objects.all()}
                edge_count = 0
                
                for edge in ksg_data.get('edges', []):
                    source_id = edge.get('source')
                    target_id = edge.get('target')
                    
                    if source_id in topic_map and target_id in topic_map:
                        GraphEdge.objects.update_or_create(
                            graph=graph,
                            source_topic=topic_map[source_id],
                            target_topic=topic_map[target_id],
                            relationship_type=edge.get('relationship', 'related'),
                            defaults={'weight': edge.get('weight', 1.0)}
                        )
                        edge_count += 1
                
                print(f"Created/updated {edge_count} graph edges")
            else:
                print(f"Knowledge graph file not found: {ksg_path}")
            
            # 6. Load questions from java_questions.yaml (optional)
            questions_path = 'static/data/java_questions.yaml'
            if os.path.exists(questions_path):
                with open(questions_path, 'r') as f:
                    questions_data = yaml.safe_load(f)
                
                # Create assessment with the correct fields
                assessment, created = Assessment.objects.get_or_create(
                    assessment_id='JAVA-ASSESSMENT-001',  # Add a valid assessment_id
                    title='Java Programming Assessment',
                    defaults={
                        'assessment_type': 'quiz',  # Use valid assessment_type from choices
                        'course': course,
                        'date': django.utils.timezone.now(),  # Add current date
                        'proctored': False
                    }
                )
                
                if created:
                    print(f"Created assessment: {assessment.title}")
                
                # Create questions
                question_count = 0
                for q_data in questions_data.get('questions', []):
                    question_text = q_data.get('question')
                    if not question_text:
                        continue
                    
                    # Get topic if specified
                    topic = None
                    topic_id = q_data.get('topic_id')
                    if topic_id:
                        topic = Topic.objects.filter(id=topic_id).first()
                    
                    # Create question
                    question, created = Question.objects.update_or_create(
                        text=question_text[:255],  # Truncate if too long
                        defaults={
                            'assessment': assessment,
                            'topic': topic,
                            'difficulty': q_data.get('difficulty', 'medium')[:20],  # Ensure within max_length
                            'points': q_data.get('points', 5),
                            'answer_explanation': q_data.get('answer', '')
                        }
                    )
                    
                    if created:
                        question_count += 1
                
                print(f"Created {question_count} questions")
            else:
                print(f"Questions file not found: {questions_path}")
            
            # 6. Create students and interactions
            from django.core.management import call_command
            
            # Create students and interactions using the management command
            print("Creating students and interactions...")
            call_command(
                'setup_students_and_interactions',
                'CS206',
                num_students=10,
                interactions_per_student=30,
                questions_file='/app/static/data/java_questions.yaml',
                topics_file='/app/static/data/java_topics.json'
            )
            
            from core.models import Student, StudentInteraction
            print(f"Created {Student.objects.count()} students")
            print(f"Created {StudentInteraction.objects.count()} interactions")
        
        print("Successfully loaded all required data!")
        return True
    
    except Exception as e:
        print(f"Error loading required data: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    load_required_data()
