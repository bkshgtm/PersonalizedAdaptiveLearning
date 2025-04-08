import networkx as nx
from django.db import transaction
from knowledge_graph.models import KnowledgeGraph, GraphEdge
from core.models import Topic

class GraphOperations:
    def __init__(self, graph_id):
        self.graph = KnowledgeGraph.objects.get(pk=graph_id)
        self.nx_graph = self._build_networkx_graph()
        
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
