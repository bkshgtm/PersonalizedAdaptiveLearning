from django.db import models
from django.contrib.auth.models import User


class KnowledgeTracingModel(models.Model):
    """Model representing a knowledge tracing model."""
    MODEL_TYPE_CHOICES = [
        ('dkt', 'Deep Knowledge Tracing'),
        ('sakt', 'Self-Attentive Knowledge Tracing'),
        ('custom', 'Custom Model'),
    ]
    
    STATUS_CHOICES = [
        ('created', 'Created'),
        ('training', 'Training'),
        ('active', 'Active'),
        ('failed', 'Failed'),
        ('archived', 'Archived'),
    ]
    
    name = models.CharField(max_length=100)
    model_type = models.CharField(max_length=20, choices=MODEL_TYPE_CHOICES)
    description = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    created_by = models.ForeignKey(User, on_delete=models.CASCADE)
    model_path = models.CharField(max_length=255)
    hyperparameters = models.JSONField(default=dict)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='created')
    is_default = models.BooleanField(default=False)
    course = models.ForeignKey('core.Course', on_delete=models.CASCADE, related_name='ml_models')
    
    def __str__(self):
        return f"{self.name} ({self.model_type})"
    
    class Meta:
        ordering = ['-updated_at']


class TrainingJob(models.Model):
    """Model representing a training job for a knowledge tracing model."""
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('processing', 'Processing'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
    ]
    
    model = models.ForeignKey(KnowledgeTracingModel, on_delete=models.CASCADE, related_name='training_jobs')
    created_at = models.DateTimeField(auto_now_add=True)
    started_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    hyperparameters = models.JSONField(default=dict)
    metrics = models.JSONField(default=dict)
    error_message = models.TextField(blank=True)
    
    # Training data information
    total_students = models.IntegerField(default=0)
    total_interactions = models.IntegerField(default=0)
    split_ratio = models.FloatField(default=0.8, help_text="Train/test split ratio")
    
    def __str__(self):
        return f"Training Job {self.id} - {self.model}"
    
    class Meta:
        ordering = ['-created_at']


class PredictionBatch(models.Model):
    """Model representing a batch of predictions."""
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('processing', 'Processing'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
    ]
    
    model = models.ForeignKey(KnowledgeTracingModel, on_delete=models.CASCADE, related_name='prediction_batches')
    created_at = models.DateTimeField(auto_now_add=True)
    started_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    
    # Batch information
    total_students = models.IntegerField(default=0)
    processed_students = models.IntegerField(default=0)
    error_message = models.TextField(blank=True)
    
    def __str__(self):
        return f"Prediction Batch {self.id} - {self.model}"
    
    class Meta:
        ordering = ['-created_at']


class TopicMastery(models.Model):
    """Model representing a student's mastery of a specific topic."""
    TREND_CHOICES = [
        ('improving', 'Improving'),
        ('declining', 'Declining'),
        ('stagnant', 'Stagnant'),
    ]
    
    student = models.ForeignKey('core.Student', on_delete=models.CASCADE, related_name='topic_masteries')
    topic = models.ForeignKey('core.Topic', on_delete=models.CASCADE, related_name='student_masteries')
    prediction_batch = models.ForeignKey(PredictionBatch, on_delete=models.CASCADE, related_name='masteries')
    mastery_score = models.FloatField(help_text="Mastery score between 0 and 1")
    confidence = models.FloatField(help_text="Confidence in the prediction between 0 and 1")
    predicted_at = models.DateTimeField(auto_now_add=True)
    trend = models.CharField(max_length=20, choices=TREND_CHOICES, default='stagnant')
    trend_data = models.JSONField(default=list, help_text="Historical mastery scores")
    
    def __str__(self):
        return f"{self.student} - {self.topic} - {self.mastery_score:.2f}"
    
    class Meta:
        ordering = ['-predicted_at']
        unique_together = ['student', 'topic', 'prediction_batch']