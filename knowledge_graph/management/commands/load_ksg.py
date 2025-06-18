import json
import os
from django.core.management.base import BaseCommand
from django.db import transaction
from django.conf import settings
from django.contrib.auth.models import User
from knowledge_graph.models import KnowledgeGraph, GraphEdge
from core.models import Topic, Course


class Command(BaseCommand):
    help = 'Load Knowledge Structure Graph from java_ksg.json'

    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS('Starting to load Knowledge Structure Graph...'))
        
        # Path to the JSON file
        json_file_path = os.path.join(settings.BASE_DIR, 'static', 'data', 'java_ksg.json')
        
        if not os.path.exists(json_file_path):
            self.stdout.write(self.style.ERROR(f'File not found: {json_file_path}'))
            return
        
        with transaction.atomic():
            # Load JSON data
            ksg_data = self.load_json_data(json_file_path)
            if not ksg_data:
                return
            
            # Get the CS206 course (all topics belong to this course)
            try:
                course = Course.objects.get(course_id='CS206')
                self.stdout.write(f'Found course: {course.title}')
            except Course.DoesNotExist:
                self.stdout.write(self.style.ERROR('CS206 course not found. Please run populate_users_courses first.'))
                return
            
            # Create or update topics from nodes
            topics_created, topics_updated = self.create_or_update_topics(ksg_data['nodes'], course)
            
            # Create or update the knowledge graph
            knowledge_graph = self.create_or_update_knowledge_graph(ksg_data)
            
            # Add all topics to the knowledge graph (this will be done through edges)
            # Note: The ManyToMany relationship uses 'through' GraphEdge, so we don't add directly
            
            # Create graph edges
            edges_created, edges_updated = self.create_graph_edges(ksg_data['edges'], knowledge_graph)
        
        self.stdout.write(self.style.SUCCESS(
            f'Successfully loaded KSG! '
            f'Topics: {topics_created} created, {topics_updated} updated. '
            f'Edges: {edges_created} created, {edges_updated} updated.'
        ))

    def load_json_data(self, json_file_path):
        """Load and validate JSON data"""
        try:
            with open(json_file_path, 'r') as file:
                data = json.load(file)
            self.stdout.write(f'Loaded JSON with {len(data.get("nodes", []))} nodes and {len(data.get("edges", []))} edges')
            return data
        except json.JSONDecodeError as e:
            self.stdout.write(self.style.ERROR(f'Error parsing JSON file: {e}'))
            return None
        except FileNotFoundError:
            self.stdout.write(self.style.ERROR(f'File not found: {json_file_path}'))
            return None

    def create_or_update_topics(self, nodes, course):
        """Create or update topics from JSON nodes"""
        topics_created = 0
        topics_updated = 0
        
        for node in nodes:
            topic_name = node.get('name')
            topic_description = node.get('description', '')
            
            if not topic_name:
                self.stdout.write(self.style.WARNING(f'Skipping node with missing name: {node}'))
                continue
            
            # Use name-based lookup to avoid ID issues (as requested)
            topic, created = Topic.objects.get_or_create(
                name=topic_name,
                course=course,
                defaults={'description': topic_description}
            )
            
            if created:
                topics_created += 1
                self.stdout.write(f'Created topic: {topic_name}')
            else:
                # Update description if it has changed
                if topic.description != topic_description:
                    topic.description = topic_description
                    topic.save()
                    topics_updated += 1
                    self.stdout.write(f'Updated topic: {topic_name}')
                else:
                    self.stdout.write(f'Topic already exists (no changes): {topic_name}')
        
        return topics_created, topics_updated

    def create_or_update_knowledge_graph(self, ksg_data):
        """Create or update the knowledge graph"""
        graph_name = ksg_data.get('name', 'Java Knowledge Structure Graph')
        graph_version = ksg_data.get('version', '1.0')
        graph_description = ksg_data.get('description', '')
        
        # Get or create a default user for the graph (required field)
        # In production, this should be the actual user creating the graph
        default_user, _ = User.objects.get_or_create(
            username='system',
            defaults={
                'email': 'system@example.com',
                'first_name': 'System',
                'last_name': 'User'
            }
        )
        
        # Create or update knowledge graph
        knowledge_graph, created = KnowledgeGraph.objects.get_or_create(
            name=graph_name,
            version=graph_version,
            defaults={
                'description': graph_description,
                'created_by': default_user,
                'data': ksg_data,  # Store the full JSON data
                'is_active': True
            }
        )
        
        if created:
            self.stdout.write(f'Created knowledge graph: {graph_name} v{graph_version}')
        else:
            # Update the data and description
            knowledge_graph.description = graph_description
            knowledge_graph.data = ksg_data
            knowledge_graph.save()
            self.stdout.write(f'Updated knowledge graph: {graph_name} v{graph_version}')
        
        return knowledge_graph

    def create_graph_edges(self, edges_data, knowledge_graph):
        """Create graph edges from JSON data"""
        edges_created = 0
        edges_updated = 0
        
        # Get the course for topic lookups
        course = Course.objects.get(course_id='CS206')
        
        # Create a mapping from node IDs to topic names for lookup
        # We need to map the JSON node IDs to actual topic names
        node_id_to_name = {}
        for node in knowledge_graph.data['nodes']:
            node_id_to_name[node['id']] = node['name']
        
        for edge in edges_data:
            try:
                # Get source and target node IDs
                source_id = edge.get('from')
                target_id = edge.get('to')
                relationship = edge.get('relationship', 'related')
                weight = edge.get('weight', 1.0)
                
                # Map IDs to topic names
                source_name = node_id_to_name.get(source_id)
                target_name = node_id_to_name.get(target_id)
                
                if not source_name or not target_name:
                    self.stdout.write(self.style.WARNING(
                        f'Skipping edge: could not find topic names for IDs {source_id} -> {target_id}'
                    ))
                    continue
                
                # Lookup topics by name
                try:
                    source_topic = Topic.objects.get(name=source_name, course=course)
                    target_topic = Topic.objects.get(name=target_name, course=course)
                except Topic.DoesNotExist as e:
                    self.stdout.write(self.style.WARNING(
                        f'Skipping edge: topic not found - {source_name} -> {target_name}: {e}'
                    ))
                    continue
                
                # Create or update graph edge
                # Note: unique_together constraint prevents duplicates
                edge_obj, created = GraphEdge.objects.get_or_create(
                    graph=knowledge_graph,
                    source_topic=source_topic,
                    target_topic=target_topic,
                    relationship_type=relationship,
                    defaults={'weight': weight}
                )
                
                if created:
                    edges_created += 1
                    self.stdout.write(f'Created edge: {source_name} -> {target_name} ({relationship})')
                else:
                    # Update weight if it has changed
                    if edge_obj.weight != weight:
                        edge_obj.weight = weight
                        edge_obj.save()
                        edges_updated += 1
                        self.stdout.write(f'Updated edge: {source_name} -> {target_name} ({relationship})')
                    else:
                        self.stdout.write(f'Edge already exists (no changes): {source_name} -> {target_name} ({relationship})')
                
            except Exception as e:
                self.stdout.write(self.style.ERROR(f'Error creating edge {edge}: {e}'))
                continue
        
        return edges_created, edges_updated
