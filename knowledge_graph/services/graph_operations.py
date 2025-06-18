import networkx as nx
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from django.db import transaction
from knowledge_graph.models import KnowledgeGraph, GraphEdge
from core.models import Topic
from ml_models.models import TopicMastery

logger = logging.getLogger(__name__)

class GraphOperations:
    def __init__(self, graph_id):
        self.graph = KnowledgeGraph.objects.get(pk=graph_id)
        
        # Build both topic-level and fine-grained graphs
        self.topic_graph = self._build_topic_graph()
        self.fine_grained_graph = self._build_fine_grained_graph()
        
        # Load KSG mapping for bridging 90 nodes to 13 topics
        self.ksg_mapping = self._load_ksg_mapping()
        self.reverse_mapping = self._build_reverse_mapping()
        
        # Legacy support
        self.nx_graph = self.topic_graph
    
    def _load_ksg_mapping(self) -> Dict[int, int]:
        """Load KSG node to topic mapping from JSON file."""
        # For our current setup, KSG nodes map directly to topics (1:1 mapping)
        # Since we have 20 topics and 20 KSG nodes with matching IDs
        mapping = {}
        if self.graph.data and 'nodes' in self.graph.data:
            for node in self.graph.data['nodes']:
                # Direct mapping: KSG node ID -> Topic ID (same ID)
                mapping[node['id']] = node['id']
        return mapping
    
    def _build_reverse_mapping(self) -> Dict[int, List[int]]:
        """Build reverse mapping: topic_id -> [ksg_node_ids]."""
        reverse_map = {}
        for ksg_node, topic_id in self.ksg_mapping.items():
            if topic_id not in reverse_map:
                reverse_map[topic_id] = []
            reverse_map[topic_id].append(ksg_node)
        return reverse_map
    
    def _build_topic_graph(self) -> nx.DiGraph:
        """Build NetworkX graph from topic-level GraphEdges (13 topics)."""
        G = nx.DiGraph()
        
        # Add all topics as nodes
        edges = GraphEdge.objects.filter(graph=self.graph)
        topic_ids = set()
        for edge in edges:
            topic_ids.add(edge.source_topic.id)
            topic_ids.add(edge.target_topic.id)

        topics = Topic.objects.filter(id__in=topic_ids)
        for topic in topics:
            G.add_node(topic.id, name=topic.name, description=topic.description)

        # Add all topic-level edges
        for edge in edges:
            G.add_edge(
                edge.source_topic.id,
                edge.target_topic.id,
                relationship=edge.relationship_type,
                weight=edge.weight
            )

        return G
    
    def _build_fine_grained_graph(self) -> nx.DiGraph:
        """Build NetworkX graph from the original 90-node KSG."""
        G = nx.DiGraph()
        
        if not self.graph.data or 'nodes' not in self.graph.data:
            return G
        
        # Add all KSG nodes (90 nodes)
        for node in self.graph.data['nodes']:
            G.add_node(
                node['id'], 
                name=node['name'],
                description=node['description']
            )
        
        # Add all KSG edges
        for edge in self.graph.data['edges']:
            G.add_edge(
                edge['from'],
                edge['to'],
                relationship=edge.get('relationship', 'related'),
                weight=edge.get('weight', 1.0)
            )
        
        return G

    def _build_networkx_graph(self):
        """Build NetworkX graph from database edges"""
        G = nx.DiGraph()
        
        # Add all topics as nodes (both source and target topics)
        edges = GraphEdge.objects.filter(graph=self.graph)
        topic_ids = set()
        for edge in edges:
            topic_ids.add(edge.source_topic.id)
            topic_ids.add(edge.target_topic.id)

        topics = Topic.objects.filter(id__in=topic_ids)
        for topic in topics:
            G.add_node(topic.id, name=topic.name, description=topic.description)

        # Add all edges
        for edge in edges:
            G.add_edge(
                edge.source_topic.id,
                edge.target_topic.id,
                relationship=edge.relationship_type,
                weight=edge.weight
            )

        return G

    def get_sample_path(self):
        """Get a sample path from a root node to any reachable node."""
        try:
            if not self.nx_graph or self.nx_graph.number_of_nodes() == 0:
                return Topic.objects.none()

            # Find root nodes (in-degree 0)
            root_nodes = [node for node, degree in self.nx_graph.in_degree() if degree == 0]
            if not root_nodes:
                # If no strict root, pick any node as a starting point (less ideal)
                root_nodes = list(self.nx_graph.nodes())
                if not root_nodes:
                    return Topic.objects.none() # Empty graph

            start_node = root_nodes[0] # Pick the first root node

            # Find any node reachable from the start_node
            reachable_nodes = list(nx.descendants(self.nx_graph, start_node))
            
            if not reachable_nodes:
                 # If the root node has no descendants, the path is just the root node itself
                 return Topic.objects.filter(id=start_node)

            # Find a simple path to the first reachable node
            # Using simple_paths avoids cycles and finds *a* path, not necessarily shortest/longest
            target_node = reachable_nodes[0]
            try:
                # Get the first path found by the generator
                path_generator = nx.all_simple_paths(self.nx_graph, source=start_node, target=target_node)
                sample_path_ids = next(path_generator)

                # Fetch topics unordered from DB
                topics_in_path = Topic.objects.filter(id__in=sample_path_ids)
                
                # Create a mapping from ID to Topic object
                topic_map = {topic.id: topic for topic in topics_in_path}
                
                # Reorder topics based on the path IDs
                ordered_topics = [topic_map[topic_id] for topic_id in sample_path_ids if topic_id in topic_map]
                
                return ordered_topics # Return the ordered list
            except (nx.NetworkXNoPath, nx.NodeNotFound, StopIteration):
                 # Fallback if path finding fails unexpectedly, return just the start node
                 return Topic.objects.filter(id=start_node)
                 
        except nx.NetworkXError:
            return Topic.objects.none()

    def get_recommended_paths(self, known_topics, target_topic_id=None):
        """Generate recommended learning paths based on known topics"""
        recommendations = []
        
        if target_topic_id:
            # Find path to specific target
            try:
                path = nx.shortest_path(self.nx_graph, source=known_topics[-1], target=target_topic_id)
                recommendations.append(Topic.objects.filter(id__in=path))
            except nx.NetworkXNoPath:
                pass
        else:
            # Recommend next topics based on mastery
            for node in self.nx_graph.nodes():
                if node not in known_topics:
                    has_prereqs = all(p in known_topics for p in self.nx_graph.predecessors(node))
                    if has_prereqs:
                        recommendations.append(Topic.objects.get(id=node))
                        
        return recommendations

    @transaction.atomic
    def get_next_topics(self, current_topic_ids):
        """Get the next recommended topics based on current knowledge"""
        if not isinstance(current_topic_ids, (list, tuple)):
            current_topic_ids = [current_topic_ids]
            
        if not current_topic_ids:
            # If no current topics, return root nodes
            root_nodes = [node for node, degree in self.nx_graph.in_degree() if degree == 0]
            return Topic.objects.filter(id__in=root_nodes)
            
        # Get all immediate successors of current topics that aren't already known
        next_topics = set()
        for topic_id in current_topic_ids:
            try:
                successors = list(self.nx_graph.successors(topic_id))
                for succ_id in successors:
                    # Only include if all prerequisites are met
                    prereqs = list(self.nx_graph.predecessors(succ_id))
                    if all(p in current_topic_ids for p in prereqs):
                        next_topics.add(succ_id)
            except nx.NetworkXError:
                continue
                    
        return Topic.objects.filter(id__in=next_topics)

    def get_prerequisites(self, topic_id, direct_only=True):
        """Get prerequisite topics for a given topic.
        
        Args:
            topic_id: ID of the target topic
            direct_only: If True, only return direct prerequisites
                        If False, return all transitive prerequisites
                        
        Returns:
            List of prerequisite topic IDs
        """
        try:
            if direct_only:
                # Get only direct prerequisites
                return list(self.nx_graph.predecessors(topic_id))
            else:
                # Get all ancestors (transitive prerequisites)
                return list(nx.ancestors(self.nx_graph, topic_id))
        except nx.NetworkXError:
            return []

    def get_detailed_prerequisites(
        self, 
        topic_id: int, 
        student_masteries: Dict[int, float] = None
    ) -> Dict[str, Any]:
        """
        Get detailed prerequisite analysis using both topic-level and fine-grained KSG.
        
        Args:
            topic_id: Target topic ID
            student_masteries: Dict mapping topic_id -> mastery_score (0-1)
            
        Returns:
            Dictionary with detailed prerequisite information
        """
        result = {
            'topic_level_prereqs': [],
            'fine_grained_prereqs': [],
            'missing_prereqs': [],
            'prerequisite_strength': 0.0,
            'recommended_subtopics': []
        }
        
        # Get topic-level prerequisites
        topic_prereqs = self.get_prerequisites(topic_id, direct_only=True)
        result['topic_level_prereqs'] = topic_prereqs
        
        # Get fine-grained prerequisites using KSG mapping
        if topic_id in self.reverse_mapping:
            ksg_nodes = self.reverse_mapping[topic_id]
            fine_grained_prereqs = set()
            
            for ksg_node in ksg_nodes:
                if ksg_node in self.fine_grained_graph:
                    # Get prerequisites for this KSG node
                    node_prereqs = list(self.fine_grained_graph.predecessors(ksg_node))
                    fine_grained_prereqs.update(node_prereqs)
            
            # Map fine-grained prerequisites back to topics
            prereq_topics = set()
            for fg_node in fine_grained_prereqs:
                if fg_node in self.ksg_mapping:
                    prereq_topic = self.ksg_mapping[fg_node]
                    if prereq_topic != topic_id:  # Don't include self
                        prereq_topics.add(prereq_topic)
            
            result['fine_grained_prereqs'] = list(prereq_topics)
        
        # Analyze missing prerequisites if student masteries provided
        if student_masteries:
            missing_prereqs = []
            prerequisite_scores = []
            
            all_prereqs = set(result['topic_level_prereqs'] + result['fine_grained_prereqs'])
            for prereq_topic in all_prereqs:
                mastery = student_masteries.get(prereq_topic, 0.0)
                prerequisite_scores.append(mastery)
                
                if mastery < 0.6:  # Threshold for "missing"
                    missing_prereqs.append({
                        'topic_id': prereq_topic,
                        'mastery_score': mastery,
                        'gap': 0.6 - mastery
                    })
            
            result['missing_prereqs'] = missing_prereqs
            result['prerequisite_strength'] = sum(prerequisite_scores) / len(prerequisite_scores) if prerequisite_scores else 0.0
        
        return result
    
    def get_learning_path_with_subtopics(
        self, 
        target_topic_id: int,
        student_masteries: Dict[int, float] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate a learning path that includes both topics and relevant subtopics.
        
        Args:
            target_topic_id: Target topic to learn
            student_masteries: Current mastery levels for topics
            
        Returns:
            List of learning path items with subtopic details
        """
        path_items = []
        
        # Get prerequisite analysis
        prereq_analysis = self.get_detailed_prerequisites(target_topic_id, student_masteries)
        
        # Process missing prerequisites first
        for missing_prereq in prereq_analysis['missing_prereqs']:
            topic_id = missing_prereq['topic_id']
            
            # Get relevant subtopics for this prerequisite
            subtopics = self._get_relevant_subtopics(topic_id, missing_prereq['mastery_score'])
            
            path_items.append({
                'topic_id': topic_id,
                'type': 'prerequisite',
                'mastery_score': missing_prereq['mastery_score'],
                'priority': 1.0 - missing_prereq['mastery_score'],  # Lower mastery = higher priority
                'subtopics': subtopics,
                'reason': f"Prerequisite for {self._get_topic_name(target_topic_id)}"
            })
        
        # Add the target topic with its subtopics
        target_mastery = student_masteries.get(target_topic_id, 0.0) if student_masteries else 0.0
        target_subtopics = self._get_relevant_subtopics(target_topic_id, target_mastery)
        
        path_items.append({
            'topic_id': target_topic_id,
            'type': 'target',
            'mastery_score': target_mastery,
            'priority': 1.0 - target_mastery,
            'subtopics': target_subtopics,
            'reason': "Target topic to learn"
        })
        
        # Sort by priority (highest first)
        path_items.sort(key=lambda x: x['priority'], reverse=True)
        
        return path_items
    
    def _get_relevant_subtopics(self, topic_id: int, mastery_score: float) -> List[Dict[str, Any]]:
        """Get relevant KSG subtopics for a topic based on mastery level."""
        subtopics = []
        
        if topic_id not in self.reverse_mapping:
            return subtopics
        
        ksg_nodes = self.reverse_mapping[topic_id]
        
        for ksg_node in ksg_nodes:
            if ksg_node in self.fine_grained_graph.nodes:
                node_data = self.fine_grained_graph.nodes[ksg_node]
                
                # Determine if this subtopic is relevant based on mastery
                relevance_score = self._calculate_subtopic_relevance(ksg_node, mastery_score)
                
                if relevance_score > 0.3:  # Threshold for relevance
                    subtopics.append({
                        'ksg_node_id': ksg_node,
                        'name': node_data.get('name', f'Subtopic {ksg_node}'),
                        'description': node_data.get('description', ''),
                        'relevance_score': relevance_score,
                        'prerequisites': list(self.fine_grained_graph.predecessors(ksg_node))
                    })
        
        # Sort by relevance score
        subtopics.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        return subtopics
    
    def _calculate_subtopic_relevance(self, ksg_node: int, topic_mastery: float) -> float:
        """Calculate how relevant a subtopic is based on current mastery."""
        # Basic relevance calculation - can be enhanced
        base_relevance = 1.0 - topic_mastery  # Lower mastery = higher relevance
        
        # Boost relevance for foundational subtopics (those with many successors)
        successors = list(self.fine_grained_graph.successors(ksg_node))
        foundation_boost = min(len(successors) * 0.1, 0.3)
        
        # Reduce relevance for advanced subtopics if mastery is low
        predecessors = list(self.fine_grained_graph.predecessors(ksg_node))
        if len(predecessors) > 3 and topic_mastery < 0.4:
            advanced_penalty = 0.2
        else:
            advanced_penalty = 0.0
        
        relevance = base_relevance + foundation_boost - advanced_penalty
        return max(0.0, min(1.0, relevance))
    
    def _get_topic_name(self, topic_id: int) -> str:
        """Get topic name by ID."""
        try:
            topic = Topic.objects.get(id=topic_id)
            return topic.name
        except Topic.DoesNotExist:
            return f"Topic {topic_id}"
    
    def get_enhanced_next_topics(
        self, 
        student_masteries: Dict[int, float],
        mastery_threshold: float = 0.6
    ) -> List[Dict[str, Any]]:
        """
        Get next recommended topics using enhanced analysis.
        
        Args:
            student_masteries: Current mastery levels
            mastery_threshold: Minimum mastery to consider a topic "learned"
            
        Returns:
            List of recommended topics with detailed reasoning
        """
        recommendations = []
        
        # Get all topics that are ready to learn (prerequisites met)
        all_topics = Topic.objects.all()
        
        for topic in all_topics:
            current_mastery = student_masteries.get(topic.id, 0.0)
            
            # Skip if already mastered
            if current_mastery >= mastery_threshold:
                continue
            
            # Get prerequisite analysis
            prereq_analysis = self.get_detailed_prerequisites(topic.id, student_masteries)
            
            # Check if prerequisites are met
            prereq_strength = prereq_analysis['prerequisite_strength']
            missing_count = len(prereq_analysis['missing_prereqs'])
            
            # Calculate readiness score
            readiness_score = self._calculate_readiness_score(
                current_mastery, prereq_strength, missing_count
            )
            
            if readiness_score > 0.3:  # Threshold for recommendation
                recommendations.append({
                    'topic_id': topic.id,
                    'topic_name': topic.name,
                    'current_mastery': current_mastery,
                    'readiness_score': readiness_score,
                    'prerequisite_strength': prereq_strength,
                    'missing_prerequisites': missing_count,
                    'detailed_analysis': prereq_analysis
                })
        
        # Sort by readiness score
        recommendations.sort(key=lambda x: x['readiness_score'], reverse=True)
        
        return recommendations
    
    def _calculate_readiness_score(
        self, 
        current_mastery: float, 
        prereq_strength: float, 
        missing_count: int
    ) -> float:
        """Calculate how ready a student is to learn a topic."""
        # Base readiness from prerequisite strength
        base_readiness = prereq_strength
        
        # Penalty for missing prerequisites
        missing_penalty = missing_count * 0.2
        
        # Boost for topics that are partially learned
        partial_learning_boost = current_mastery * 0.3
        
        readiness = base_readiness - missing_penalty + partial_learning_boost
        return max(0.0, min(1.0, readiness))

    def import_from_json(self, json_data):
        """Import graph structure from JSON data"""
        # Clear existing edges
        GraphEdge.objects.filter(graph=self.graph).delete()
        
        # Create new edges
        for edge in json_data['edges']:
            GraphEdge.objects.create(
                graph=self.graph,
                source_topic_id=edge['source'],
                target_topic_id=edge['target'],
                relationship_type=edge['relationship'],
                weight=edge.get('weight', 1.0)
            )
        
        # Update graph data
        self.graph.data = json_data
        self.graph.save()
