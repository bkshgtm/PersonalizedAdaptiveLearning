from django.db import models
from django.contrib.auth.models import User
import datetime
import json


class LearningPath(models.Model):
    """Model representing a generated learning path for a student."""
    STATUS_CHOICES = [
        ('active', 'Active'),
        ('completed', 'Completed'),
        ('archived', 'Archived'),
    ]
    
    # Core identification
    student = models.ForeignKey('core.Student', on_delete=models.CASCADE, related_name='learning_paths')
    course = models.ForeignKey('core.Course', on_delete=models.CASCADE, related_name='learning_paths')
    
    # Path metadata
    name = models.CharField(max_length=200, default="Personalized Learning Path")
    description = models.TextField(blank=True)
    generated_at = models.DateTimeField(auto_now_add=True)
    expires_at = models.DateTimeField(null=True, blank=True)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='active')
    
    # Student stats (stored as JSON for flexibility)
    student_stats = models.JSONField(default=dict, help_text="Student profile and performance statistics")
    
    # Path summary
    total_estimated_time = models.FloatField(default=0.0, help_text="Total estimated learning time in hours")
    weak_topics_count = models.IntegerField(default=0)
    recommended_topics_count = models.IntegerField(default=0)
    
    # Progress tracking
    overall_progress = models.FloatField(default=0.0, help_text="Overall completion percentage (0-100)")
    last_accessed = models.DateTimeField(null=True, blank=True)
    
    def __str__(self):
        return f"Learning Path for {self.student.student_id} - {self.name}"
    
    class Meta:
        ordering = ['-generated_at']
        unique_together = ['student', 'course', 'generated_at']


class WeakTopic(models.Model):
    """Model representing a weak topic identified for a student."""
    learning_path = models.ForeignKey(LearningPath, on_delete=models.CASCADE, related_name='weak_topics')
    topic = models.ForeignKey('core.Topic', on_delete=models.CASCADE)
    
    # Mastery information
    current_mastery = models.FloatField(help_text="Current mastery score (0-1)")
    
    # Prerequisites and relationships
    prerequisites = models.JSONField(default=list, help_text="List of prerequisite topic names")
    related_topics = models.JSONField(default=list, help_text="List of related topic names")
    
    # Order in the weak topics list
    order = models.PositiveIntegerField(default=0)
    
    def __str__(self):
        return f"{self.learning_path.student.student_id} - Weak: {self.topic.name}"
    
    class Meta:
        ordering = ['learning_path', 'order']
        unique_together = ['learning_path', 'topic']


class RecommendedTopic(models.Model):
    """Model representing a recommended topic in the learning path."""
    learning_path = models.ForeignKey(LearningPath, on_delete=models.CASCADE, related_name='recommended_topics')
    topic = models.ForeignKey('core.Topic', on_delete=models.CASCADE)
    
    # Recommendation details
    confidence = models.FloatField(help_text="Confidence score for this recommendation (0-1)")
    recommended_difficulty = models.CharField(max_length=20, default='intermediate')
    estimated_time_hours = models.FloatField(help_text="Estimated time to complete in hours")
    
    # Prerequisites
    prerequisites = models.JSONField(default=list, help_text="List of prerequisite topic names")
    unmet_prerequisites = models.JSONField(default=list, help_text="List of unmet prerequisite topic names")
    should_study_prerequisites_first = models.BooleanField(default=False)
    
    # Relationships
    related_topics = models.JSONField(default=list, help_text="List of related topic names")
    
    # Order in the recommended path
    priority = models.PositiveIntegerField(help_text="Priority order (1 = highest priority)")
    
    # Progress tracking
    completed = models.BooleanField(default=False)
    completed_at = models.DateTimeField(null=True, blank=True)
    started_at = models.DateTimeField(null=True, blank=True)
    progress_percentage = models.FloatField(default=0.0, help_text="Completion percentage (0-100)")
    
    def __str__(self):
        return f"{self.learning_path.student.student_id} - Rec: {self.topic.name} (Priority {self.priority})"
    
    class Meta:
        ordering = ['learning_path', 'priority']
        unique_together = ['learning_path', 'topic']


class TopicResource(models.Model):
    """Model representing a resource recommended for a specific topic in a learning path."""
    # Can be linked to either weak topics or recommended topics
    weak_topic = models.ForeignKey(WeakTopic, on_delete=models.CASCADE, related_name='resources', null=True, blank=True)
    recommended_topic = models.ForeignKey(RecommendedTopic, on_delete=models.CASCADE, related_name='resources', null=True, blank=True)
    
    # Resource details (stored directly for historical preservation)
    title = models.CharField(max_length=300)
    description = models.TextField()
    url = models.URLField(max_length=500)
    resource_type = models.CharField(max_length=50)  # video, documentation, visual, other, etc.
    difficulty = models.CharField(max_length=20)     # beginner, intermediate, advanced
    estimated_time = models.FloatField(help_text="Estimated time in hours")
    
    # Order in the resource list for this topic
    order = models.PositiveIntegerField(default=0)
    
    # Progress tracking
    viewed = models.BooleanField(default=False)
    viewed_at = models.DateTimeField(null=True, blank=True)
    completed = models.BooleanField(default=False)
    completed_at = models.DateTimeField(null=True, blank=True)
    rating = models.IntegerField(null=True, blank=True, help_text="User rating 1-5")
    notes = models.TextField(blank=True, help_text="Student notes about this resource")
    
    def __str__(self):
        topic_name = self.weak_topic.topic.name if self.weak_topic else self.recommended_topic.topic.name
        return f"{topic_name} - {self.title}"
    
    class Meta:
        ordering = ['order']


class LearningPathProgress(models.Model):
    """Model tracking overall progress on a learning path."""
    learning_path = models.OneToOneField(LearningPath, on_delete=models.CASCADE, related_name='progress_detail')
    
    # Time tracking
    total_time_spent = models.DurationField(default=datetime.timedelta(0))
    last_study_session = models.DateTimeField(null=True, blank=True)
    study_streak_days = models.IntegerField(default=0)
    
    # Topic progress
    topics_started = models.IntegerField(default=0)
    topics_completed = models.IntegerField(default=0)
    resources_viewed = models.IntegerField(default=0)
    resources_completed = models.IntegerField(default=0)
    
    # Performance metrics
    average_session_time = models.DurationField(default=datetime.timedelta(0))
    completion_rate = models.FloatField(default=0.0, help_text="Percentage of started topics completed")
    
    # Milestones
    milestones_achieved = models.JSONField(default=list, help_text="List of achieved milestone names")
    next_milestone = models.CharField(max_length=200, blank=True)
    
    def __str__(self):
        return f"Progress for {self.learning_path.student.student_id}"
    
    class Meta:
        verbose_name = "Learning Path Progress"
        verbose_name_plural = "Learning Path Progress"


class StudySession(models.Model):
    """Model representing a study session for a learning path."""
    learning_path = models.ForeignKey(LearningPath, on_delete=models.CASCADE, related_name='study_sessions')
    
    # Session details
    started_at = models.DateTimeField()
    ended_at = models.DateTimeField(null=True, blank=True)
    duration = models.DurationField(null=True, blank=True)
    
    # What was studied
    topics_studied = models.ManyToManyField('core.Topic', related_name='study_sessions')
    resources_accessed = models.JSONField(default=list, help_text="List of resource IDs accessed")
    
    # Session outcomes
    topics_completed = models.IntegerField(default=0)
    resources_completed = models.IntegerField(default=0)
    notes = models.TextField(blank=True, help_text="Student notes from this session")
    
    # Session quality
    effectiveness_rating = models.IntegerField(null=True, blank=True, help_text="Self-rated effectiveness 1-5")
    difficulty_rating = models.IntegerField(null=True, blank=True, help_text="Self-rated difficulty 1-5")
    
    def __str__(self):
        return f"Study Session - {self.learning_path.student.student_id} - {self.started_at.date()}"
    
    class Meta:
        ordering = ['-started_at']


class PathFeedback(models.Model):
    """Model for collecting feedback on learning paths."""
    FEEDBACK_TYPE_CHOICES = [
        ('helpful', 'Helpful'),
        ('not_helpful', 'Not Helpful'),
        ('too_easy', 'Too Easy'),
        ('too_hard', 'Too Hard'),
        ('wrong_order', 'Wrong Order'),
        ('missing_topics', 'Missing Topics'),
        ('other', 'Other'),
    ]
    
    learning_path = models.ForeignKey(LearningPath, on_delete=models.CASCADE, related_name='feedback')
    feedback_type = models.CharField(max_length=20, choices=FEEDBACK_TYPE_CHOICES)
    
    # Specific feedback
    topic = models.ForeignKey('core.Topic', on_delete=models.CASCADE, null=True, blank=True, 
                             help_text="Specific topic this feedback relates to")
    resource = models.ForeignKey(TopicResource, on_delete=models.CASCADE, null=True, blank=True,
                                help_text="Specific resource this feedback relates to")
    
    # Feedback content
    rating = models.IntegerField(help_text="Rating 1-5")
    comment = models.TextField(blank=True)
    
    # Metadata
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"Feedback - {self.learning_path.student.student_id} - {self.feedback_type}"
    
    class Meta:
        ordering = ['-created_at']
