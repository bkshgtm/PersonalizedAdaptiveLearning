import json
import logging
from typing import Dict, List, Any, Optional

from .graph_operations import GraphOperations

logger = logging.getLogger(__name__)


class GraphVisualizer:
    """
    Service class for visualizing the knowledge graph.
    """
    
    def __init__(self, graph_operations: GraphOperations):
        """
        Initialize with a GraphOperations instance.
        """
        self.graph_ops = graph_operations
    
    def get_d3_format(self) -> Dict[str, Any]:
        """
        Get graph data in D3.js format for visualization.
        
        Returns:
            Dictionary with nodes and links in D3.js format
        """
        nodes = []
        links = []
        
        # Get nodes
        for node_id, data in self.graph_ops.nx_graph.nodes(data=True):
            nodes.append({
                'id': node_id,
                'name': data.get('name', ''),
                'description': data.get('description', ''),
                'parent_id': data.get('parent_id')
            })
        
        # Get links
        for source, target, data in self.graph_ops.nx_graph.edges(data=True):
            links.append({
                'source': source,
                'target': target,
                'type': data.get('relationship', 'related'),
                'weight': data.get('weight', 1.0)
            })
        
        return {
            'nodes': nodes,
            'links': links
        }
    
    def get_vis_js_format(self) -> Dict[str, Any]:
        """
        Get graph data in vis.js format for visualization.
        
        Returns:
            Dictionary with nodes and edges in vis.js format
        """
        nodes = []
        edges = []
        
        # Get nodes
        for node_id, data in self.graph_ops.nx_graph.nodes(data=True):
            nodes.append({
                'id': node_id,
                'label': data.get('name', ''),
                'title': data.get('description', ''),
                'group': 'topic' if data.get('parent_id') is None else 'subtopic'
            })
        
        # Get edges
        edge_id = 0
        for source, target, data in self.graph_ops.nx_graph.edges(data=True):
            relationship = data.get('relationship', 'related')
            edges.append({
                'id': edge_id,
                'from': source,
                'to': target,
                'label': relationship,
                'arrows': 'to',
                'color': self._get_edge_color(relationship),
                'width': data.get('weight', 1.0) * 2
            })
            edge_id += 1
        
        return {
            'nodes': nodes,
            'edges': edges
        }
    
    def get_learning_path_visualization(self, mastered_topics: List[int], target_topics: List[int]) -> Dict[str, Any]:
        """
        Get visualization data for a learning path.
        
        Args:
            mastered_topics: List of topic IDs that the student has mastered
            target_topics: List of target topic IDs to learn
        
        Returns:
            Dictionary with nodes and edges for visualization
        """
        # Get the learning path
        path = self.graph_ops.find_learning_path(mastered_topics, target_topics)
        path_ids = [topic['id'] for topic in path]
        
        # Get graph in vis.js format
        graph_data = self.get_vis_js_format()
        
        # Mark nodes in the path
        for node in graph_data['nodes']:
            node_id = node['id']
            if node_id in mastered_topics:
                node['group'] = 'mastered'
                node['color'] = '#4CAF50'  # Green
            elif node_id in target_topics:
                node['group'] = 'target'
                node['color'] = '#F44336'  # Red
            elif node_id in path_ids:
                node['group'] = 'path'
                node['color'] = '#2196F3'  # Blue
        
        # Mark edges in the path
        for i in range(len(path) - 1):
            current_id = path[i]['id']
            next_id = path[i + 1]['id']
            
            for edge in graph_data['edges']:
                if edge['from'] == current_id and edge['to'] == next_id:
                    edge['color'] = '#2196F3'  # Blue
                    edge['width'] = 4
        
        return graph_data
    
    def _get_edge_color(self, relationship: str) -> Dict[str, str]:
        """
        Get the color for an edge based on its relationship type.
        
        Args:
            relationship: Type of relationship
        
        Returns:
            Dictionary with color information
        """
        if relationship == 'prerequisite':
            return {'color': '#FF5722', 'highlight': '#FF7043'}  # Orange
        elif relationship == 'next':
            return {'color': '#4CAF50', 'highlight': '#66BB6A'}  # Green
        elif relationship == 'part_of':
            return {'color': '#9C27B0', 'highlight': '#AB47BC'}  # Purple
        else:  # 'related'
            return {'color': '#2196F3', 'highlight': '#42A5F5'}  # Blue