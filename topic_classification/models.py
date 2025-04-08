from django.db import models
from django.contrib.auth.models import User


class ClassificationModel(models.Model):
    """Model representing a topic classification model."""
    MODEL_TYPE_CHOICES = [
        ('tfidf', 'TF-IDF + Cosine Similarity'),
        ('transformer', 'Transformer'),
        ('openai', 'OpenAI API'),
        ('anthropic', 'Anthropic API'),
        ('dissect', 'Dissect API'),
        ('custom', 'Custom Model'),
    ]
    
    # Rest of the model remains the same...
    
    STATUS_CHOICES = [
        ('active', 'Active'),
        ('inactive', 'Inactive'),
        ('training', 'Training'),
        ('failed', 'Failed'),
    ]
    
    name = models.CharField(max_length=100)
    model_type = models.CharField(max_length=20, choices=MODEL_TYPE_CHOICES)
    description = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    created_by = models.ForeignKey(User, on_delete=models.CASCADE)
    model_path = models.CharField(max_length=255, blank=True)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='inactive')
    metadata = models.JSONField(default=dict, blank=True)
    is_default = models.BooleanField(default=False)
    
    def __str__(self):
        return f"{self.name} ({self.model_type})"
    
    class Meta:
        ordering = ['-updated_at']


class ClassificationJob(models.Model):
    """Model representing a batch classification job."""
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('processing', 'Processing'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
    ]
    
    model = models.ForeignKey(ClassificationModel, on_delete=models.CASCADE, related_name='jobs')
    created_at = models.DateTimeField(auto_now_add=True)
    started_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    total_questions = models.IntegerField(default=0)
    classified_questions = models.IntegerField(default=0)
    failed_questions = models.IntegerField(default=0)
    error_message = models.TextField(blank=True)
    
    def __str__(self):
        return f"Job {self.id} - {self.model} - {self.status}"
    
    class Meta:
        ordering = ['-created_at']


class ClassificationResult(models.Model):
    """Model representing a single question's classification result."""
    job = models.ForeignKey(ClassificationJob, on_delete=models.CASCADE, related_name='results')
    question = models.ForeignKey('core.Question', on_delete=models.CASCADE, related_name='classification_results')
    topic = models.ForeignKey('core.Topic', on_delete=models.CASCADE, related_name='classification_results')
    confidence = models.FloatField(help_text="Confidence score (0-1)")
    classified_at = models.DateTimeField(auto_now_add=True)
    is_verified = models.BooleanField(default=False)
    verified_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True)
    raw_output = models.JSONField(default=dict, blank=True)
    
    def __str__(self):
        return f"Classification of {self.question} as {self.topic} ({self.confidence:.2f})"
    
    class Meta:
        ordering = ['-classified_at']
