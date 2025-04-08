from django.db import models
from django.contrib.auth.models import User
import datetime


class PathGenerator(models.Model):
    """Model representing a learning path generator."""
    name = models.CharField(max_length=100)
    description = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    created_by = models.ForeignKey(User, on_delete=models.CASCADE)
    is_active = models.BooleanField(default=True)
    config = models.JSONField(default=dict, help_text="Configuration parameters for the generator")
    
    def __str__(self):
        return self.name
    
    class Meta:
        ordering = ['-updated_at']


class PathGenerationJob(models.Model):
    """Model representing a learning path generation job."""
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('processing', 'Processing'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
    ]
    
    generator = models.ForeignKey(PathGenerator, on_delete=models.CASCADE, related_name='jobs')
    student = models.ForeignKey('core.Student', on_delete=models.CASCADE, related_name='path_generation_jobs')
    course = models.ForeignKey('core.Course', on_delete=models.CASCADE, related_name='path_generation_jobs')
    created_at = models.DateTimeField(auto_now_add=True)
    started_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    error_message = models.TextField(blank=True)
    
    # Reference to the prediction batch used for knowledge states
    prediction_batch = models.ForeignKey('ml_models.PredictionBatch', on_delete=models.SET_NULL, 
                                       null=True, blank=True, related_name='path_generation_jobs')
    
    # Reference to the knowledge graph used
    knowledge_graph = models.ForeignKey('knowledge_graph.KnowledgeGraph', on_delete=models.SET_NULL,
                                      null=True, blank=True, related_name='path_generation_jobs')
    
    def __str__(self):
        return f"Path Generation Job {self.id} - {self.student.student_id}"
    
    class Meta:
        ordering = ['-created_at']


class LearningPath(models.Model):
    """Model representing a generated learning path."""
    STATUS_CHOICES = [
        ('active', 'Active'),
        ('completed', 'Completed'),
        ('archived', 'Archived'),
    ]
    
    generation_job = models.OneToOneField(PathGenerationJob, on_delete=models.CASCADE, related_name='path')
    student = models.ForeignKey('core.Student', on_delete=models.CASCADE, related_name='learning_paths')
    course = models.ForeignKey('core.Course', on_delete=models.CASCADE, related_name='learning_paths')
    name = models.CharField(max_length=200)
    description = models.TextField(blank=True)
    generated_at = models.DateTimeField(auto_now_add=True)
    expires_at = models.DateTimeField(null=True, blank=True)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='active')
    overall_progress = models.JSONField(default=dict)
    estimated_completion_time = models.DurationField(default=datetime.timedelta(hours=1))
    
    def __str__(self):
        return f"Learning Path for {self.student.student_id} - {self.name}"
    
    class Meta:
        ordering = ['-generated_at']


class LearningPathItem(models.Model):
    """Model representing an item in a learning path."""
    STATUS_CHOICES = [
        ('weak', 'Weak'),
        ('developing', 'Developing'),
        ('strong', 'Strong'),
    ]
    
    TREND_CHOICES = [
        ('improving', 'Improving'),
        ('declining', 'Declining'),
        ('stagnant', 'Stagnant'),
    ]
    
    path = models.ForeignKey(LearningPath, on_delete=models.CASCADE, related_name='items')
    topic = models.ForeignKey('core.Topic', on_delete=models.CASCADE, related_name='path_items')
    priority = models.IntegerField()
    status = models.CharField(max_length=20, choices=STATUS_CHOICES)
    proficiency_score = models.FloatField(help_text="Mastery score between 0 and 1")
    trend = models.CharField(max_length=20, choices=TREND_CHOICES)
    confidence_of_improvement = models.FloatField(help_text="Confidence in improvement prediction between 0 and 1")
    reason = models.TextField(help_text="Explanation for why this topic is in the path")
    estimated_review_time = models.DurationField(help_text="Estimated time to review this topic")
    
    # Progress tracking
    completed = models.BooleanField(default=False)
    completed_at = models.DateTimeField(null=True, blank=True)
    
    def __str__(self):
        return f"{self.path} - {self.topic.name} - Priority {self.priority}"
    
    class Meta:
        ordering = ['path', 'priority']
        unique_together = ['path', 'topic']


class LearningResource(models.Model):
    """Model representing a recommended learning resource for a path item."""
    path_item = models.ForeignKey(LearningPathItem, on_delete=models.CASCADE, related_name='resources')
    resource = models.ForeignKey('core.Resource', on_delete=models.CASCADE, related_name='path_recommendations')
    match_reason = models.TextField(help_text="Reason why this resource is recommended")
    
    # Progress tracking
    viewed = models.BooleanField(default=False)
    viewed_at = models.DateTimeField(null=True, blank=True)
    
    def __str__(self):
        return f"{self.path_item} - {self.resource.title}"
    
    class Meta:
        ordering = ['path_item']


class PathCheckpoint(models.Model):
    """Model representing a checkpoint in a learning path."""
    TYPE_CHOICES = [
        ('quiz', 'Quiz'),
        ('exercise', 'Exercise'),
        ('reflection', 'Reflection'),
    ]
    
    path = models.ForeignKey(LearningPath, on_delete=models.CASCADE, related_name='checkpoints')
    name = models.CharField(max_length=100)
    checkpoint_type = models.CharField(max_length=20, choices=TYPE_CHOICES)
    description = models.TextField()
    position = models.IntegerField(help_text="Position in the learning path")
    
    # Topics covered by this checkpoint
    topics = models.ManyToManyField('core.Topic', related_name='checkpoints')
    
    # Progress tracking
    completed = models.BooleanField(default=False)
    completed_at = models.DateTimeField(null=True, blank=True)
    score = models.FloatField(null=True, blank=True)
    
    def __str__(self):
        return f"{self.path} - {self.name}"
    
    class Meta:
        ordering = ['path', 'position']