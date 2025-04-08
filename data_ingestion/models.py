from django.db import models
from django.contrib.auth.models import User


class DataUpload(models.Model):
    """Model representing a data upload operation."""
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('processing', 'Processing'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
    ]
    
    file = models.FileField(upload_to='data_uploads/')
    uploaded_by = models.ForeignKey(User, on_delete=models.CASCADE)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    rows_processed = models.IntegerField(default=0)
    rows_failed = models.IntegerField(default=0)
    error_message = models.TextField(blank=True)
    
    def __str__(self):
        return f"Upload {self.id} - {self.file.name} - {self.status}"
    
    class Meta:
        ordering = ['-uploaded_at']


class ProcessingLog(models.Model):
    """Model to track the progress and status of data processing tasks."""
    data_upload = models.ForeignKey(DataUpload, on_delete=models.CASCADE, related_name='logs', null=True, blank=True)
    document_upload = models.ForeignKey('DocumentUpload', on_delete=models.CASCADE, related_name='logs', null=True, blank=True)
    timestamp = models.DateTimeField(auto_now_add=True)
    message = models.TextField()
    level = models.CharField(max_length=20, default='info')
    
    def __str__(self):
        if self.data_upload:
            return f"{self.data_upload} - {self.timestamp} - {self.level}"
        elif self.document_upload:
            return f"{self.document_upload} - {self.timestamp} - {self.level}"
        else:
            return f"Log - {self.timestamp} - {self.level}"
    
    class Meta:
        ordering = ['-timestamp']


class DocumentUpload(models.Model):
    """Model representing a document upload containing questions and answers."""
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('processing', 'Processing'),
        ('completed', 'Completed'),
        ('completed with errors', 'Completed with Errors'),
        ('failed', 'Failed'),
    ]
    
    DOCUMENT_TYPE_CHOICES = [
        ('pdf', 'PDF Document'),
        ('docx', 'Word Document'),
        ('txt', 'Text File'),
        ('other', 'Other Format'),
    ]
    
    file = models.FileField(upload_to='document_uploads/')
    document_type = models.CharField(max_length=10, choices=DOCUMENT_TYPE_CHOICES)
    uploaded_by = models.ForeignKey(User, on_delete=models.CASCADE)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    status = models.CharField(max_length=30, choices=STATUS_CHOICES, default='pending')
    student = models.ForeignKey('core.Student', on_delete=models.CASCADE, null=True, blank=True)
    course = models.ForeignKey('core.Course', on_delete=models.CASCADE, null=True, blank=True)
    assessment = models.ForeignKey('core.Assessment', on_delete=models.CASCADE, null=True, blank=True)
    questions_processed = models.IntegerField(default=0)
    questions_failed = models.IntegerField(default=0)
    error_message = models.TextField(blank=True)
    
    def __str__(self):
        return f"Document {self.id} - {self.file.name} - {self.status}"
    
    class Meta:
        ordering = ['-uploaded_at']


class ExtractedQuestion(models.Model):
    """Model representing a question extracted from a document."""
    document = models.ForeignKey(DocumentUpload, on_delete=models.CASCADE, related_name='questions')
    question_text = models.TextField()
    student_answer = models.TextField()
    extracted_at = models.DateTimeField(auto_now_add=True)
    question_number = models.IntegerField(default=0)
    page_number = models.IntegerField(default=1)
    is_processed = models.BooleanField(default=False)
    
    # After processing
    topic = models.ForeignKey('core.Topic', on_delete=models.SET_NULL, null=True, blank=True)
    is_correct = models.BooleanField(null=True, blank=True)
    confidence_score = models.FloatField(null=True, blank=True)
    feedback = models.TextField(blank=True)
    validation_metadata = models.JSONField(null=True, blank=True)  # New field for detailed validation results
    
    def __str__(self):
        return f"Question {self.question_number} from {self.document}"

    class Meta:
        ordering = ['document', 'question_number']
