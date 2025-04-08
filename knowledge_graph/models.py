from django.db import models
from django.contrib.auth.models import User


class KnowledgeGraph(models.Model):
    """Model representing a knowledge structure graph for topics."""
    name = models.CharField(max_length=100)
    description = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    created_by = models.ForeignKey(User, on_delete=models.CASCADE)
    version = models.CharField(max_length=20)
    data = models.JSONField(help_text="Graph structure as JSON")
    is_active = models.BooleanField(default=False)
    topics = models.ManyToManyField(
        'core.Topic',
        through='GraphEdge',
        through_fields=('graph', 'source_topic'),
        related_name='knowledge_graphs'
    )
    
    def __str__(self):
        return f"{self.name} (v{self.version})"
    
    class Meta:
        ordering = ['-updated_at']
        unique_together = ['name', 'version']


class GraphEdge(models.Model):
    """Model representing an edge in the knowledge graph."""
    RELATIONSHIP_TYPES = [
        ('prerequisite', 'Prerequisite'),
        ('related', 'Related'),
        ('part_of', 'Part Of'),
        ('next', 'Next'),
    ]
    
    graph = models.ForeignKey(KnowledgeGraph, on_delete=models.CASCADE, related_name='edges')
    source_topic = models.ForeignKey('core.Topic', on_delete=models.CASCADE, related_name='outgoing_edges')
    target_topic = models.ForeignKey('core.Topic', on_delete=models.CASCADE, related_name='incoming_edges')
    relationship_type = models.CharField(max_length=20, choices=RELATIONSHIP_TYPES)
    weight = models.FloatField(default=1.0, help_text="Edge weight (0-1)")
    
    def __str__(self):
        return f"{self.source_topic} -> {self.target_topic} ({self.relationship_type})"
    
    class Meta:
        ordering = ['graph', 'source_topic']
        unique_together = ['graph', 'source_topic', 'target_topic', 'relationship_type']
