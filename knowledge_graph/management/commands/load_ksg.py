from django.core.management.base import BaseCommand
from django.contrib.auth import get_user_model
from knowledge_graph.models import KnowledgeGraph, GraphEdge
from core.models import Topic
import json
from pathlib import Path

class Command(BaseCommand):
    help = 'Loads the initial Java Knowledge Structure Graph'

    def handle(self, *args, **options):
        User = get_user_model()
        admin_user = User.objects.filter(is_superuser=True).first()
        
        if not admin_user:
            self.stdout.write(self.style.ERROR('No admin user found'))
            return

        # Load KSG data from JSON
        ksg_path = Path('static/data/sample_ksg.json')
        try:
            with open(ksg_path) as f:
                ksg_data = json.load(f)
        except FileNotFoundError:
            self.stdout.write(self.style.ERROR(f'KSG file not found at {ksg_path}'))
            return
        except json.JSONDecodeError:
            self.stdout.write(self.style.ERROR(f'Invalid JSON in KSG file'))
            return

        # Create or update KnowledgeGraph
        graph, created = KnowledgeGraph.objects.update_or_create(
            name=ksg_data['name'],
            version=ksg_data['version'],
            defaults={
                'description': ksg_data['description'],
                'data': ksg_data,
                'is_active': True,
                'created_by': admin_user
            }
        )

        # Create topics
        topic_map = {}
        for node in ksg_data['nodes']:
            # Get the first available course
            from core.models import Course
            course = Course.objects.first()
            
            topic, _ = Topic.objects.update_or_create(
                name=node['name'],
                defaults={
                    'description': node['description'],
                    'course': course
                }
            )
            topic_map[node['id']] = topic

        # Create edges with validation
        edge_count = 0
        for edge in ksg_data['edges']:
            source_id = edge['source']
            target_id = edge['target']
            
            if source_id not in topic_map:
                self.stdout.write(self.style.WARNING(
                    f"Skipping edge - source topic {source_id} not found"
                ))
                continue
                
            if target_id not in topic_map:
                self.stdout.write(self.style.WARNING(
                    f"Skipping edge - target topic {target_id} not found"
                ))
                continue
                
            GraphEdge.objects.update_or_create(
                graph=graph,
                source_topic=topic_map[source_id],
                target_topic=topic_map[target_id],
                relationship_type=edge['relationship'],
                defaults={'weight': edge.get('weight', 1.0)}
            )
            edge_count += 1

        self.stdout.write(self.style.SUCCESS(
            f"Successfully loaded {len(topic_map)} topics and {edge_count} edges"
        ))
