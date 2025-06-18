from django.db import models
from django.contrib.auth.models import User


class Student(models.Model):
    """Model representing a student in the learning system."""
    ACADEMIC_LEVEL_CHOICES = [
        ('freshman', 'Freshman'),
        ('sophomore', 'Sophomore'),
        ('junior', 'Junior'),
        ('senior', 'Senior'),
        ('graduate', 'Graduate'),
    ]
    
    STUDY_FREQUENCY_CHOICES = [
        ('daily', 'Daily'),
        ('weekly', 'Weekly'),
        ('biweekly', 'Bi-Weekly'),
        ('monthly', 'Monthly'),
        ('rarely', 'Rarely'),
    ]
    
    student_id = models.CharField(max_length=50, primary_key=True)
    user = models.OneToOneField(User, on_delete=models.CASCADE, null=True, blank=True)
    major = models.CharField(max_length=100)
    academic_level = models.CharField(max_length=20, choices=ACADEMIC_LEVEL_CHOICES)
    gpa = models.FloatField()
    prior_knowledge_score = models.FloatField(null=True, blank=True)
    study_frequency = models.CharField(max_length=20, choices=STUDY_FREQUENCY_CHOICES)
    attendance_rate = models.FloatField(help_text="Percentage of attendance")
    participation_score = models.FloatField(help_text="Score for participation in forums, etc.")
    last_login_date = models.DateTimeField(null=True, blank=True)
    total_time_spent = models.DurationField(null=True, blank=True)
    average_time_per_session = models.DurationField(null=True, blank=True)
    
    def __str__(self):
        return f"Student {self.student_id}"
    
    class Meta:
        ordering = ['student_id']


class Course(models.Model):
    """Model representing a course."""
    course_id = models.CharField(max_length=50, primary_key=True)
    title = models.CharField(max_length=200)
    description = models.TextField(blank=True)
    students = models.ManyToManyField(Student, related_name='courses', blank=True)
    
    def __str__(self):
        return self.title
    
    class Meta:
        ordering = ['course_id']


class Topic(models.Model):
    """Model representing a topic in the course curriculum."""
    name = models.CharField(max_length=100)
    description = models.TextField(blank=True)
    parent = models.ForeignKey('self', null=True, blank=True, on_delete=models.SET_NULL,
                              related_name='subtopics')
    course = models.ForeignKey(Course, on_delete=models.CASCADE, related_name='topics')
    
    def __str__(self):
        return self.name
    
    class Meta:
        ordering = ['name']
        unique_together = ['name', 'course']


class Resource(models.Model):
    """Model representing a learning resource."""
    TYPE_CHOICES = [
        ('video', 'Video'),
        ('document', 'Document'),
        ('exercise', 'Exercise'),
        ('tutorial', 'Tutorial'),
        ('quiz', 'Quiz'),
        ('example', 'Example'),
        ('reference', 'Reference'),
        ('documentation', 'Documentation'),
        ('course', 'Course'),
        ('book', 'Book'),
        ('practice', 'Practice'),
        ('video','Video'),
        ('documentation', 'Documentation'),
        ('visual', 'Visual'),
        ('other','Other')
    ]
    
    DIFFICULTY_CHOICES = [
        ('beginner', 'Beginner'),
        ('intermediate', 'Intermediate'),
        ('advanced', 'Advanced'),
    ]
    
    title = models.CharField(max_length=200)
    description = models.TextField(blank=True)
    url = models.URLField()
    resource_type = models.CharField(max_length=50, choices=TYPE_CHOICES)
    difficulty = models.CharField(max_length=20, choices=DIFFICULTY_CHOICES)
    estimated_time = models.DurationField(null=True, blank=True, help_text="Estimated time to complete resource")
    topics = models.ManyToManyField(Topic, related_name='resources')
    
    def __str__(self):
        return self.title
    
    class Meta:
        ordering = ['title']


class Assessment(models.Model):
    """Model representing an assessment (quiz, exam, etc.)."""
    ASSESSMENT_TYPE_CHOICES = [
        ('quiz', 'Quiz'),
        ('exam', 'Exam'),
        ('assignment', 'Assignment'),
        ('project', 'Project'),
    ]
    
    assessment_id = models.CharField(max_length=50, primary_key=True)
    title = models.CharField(max_length=200)
    assessment_type = models.CharField(max_length=20, choices=ASSESSMENT_TYPE_CHOICES)
    course = models.ForeignKey(Course, on_delete=models.CASCADE, related_name='assessments')
    date = models.DateTimeField()
    proctored = models.BooleanField(default=False)
    
    def __str__(self):
        return self.title
    
    class Meta:
        ordering = ['-date']


class Question(models.Model):
    """Model representing a question in an assessment."""
    QUESTION_TYPE_CHOICES = [
        ('mcq', 'Multiple Choice'),
        ('coding', 'Coding'),
        ('code_analysis', 'Code Analysis'),
        ('fill_blank', 'Fill in the Blank'),
        ('short_answer', 'Short Answer'),
        ('essay', 'Essay'),
        ('true_false', 'True/False'),
    ]
    
    question_id = models.CharField(max_length=50, primary_key=True)
    assessment = models.ForeignKey(Assessment, on_delete=models.CASCADE, related_name='questions')
    text = models.TextField()
    question_type = models.CharField(max_length=20, choices=QUESTION_TYPE_CHOICES)
    topic = models.ForeignKey(Topic, null=True, blank=True, on_delete=models.SET_NULL, 
                             related_name='questions')
    
    def __str__(self):
        return f"Question {self.question_id}"
    
    class Meta:
        ordering = ['question_id']


class StudentInteraction(models.Model):
    """Model representing a student's interaction with a question."""
    student = models.ForeignKey(Student, on_delete=models.CASCADE, related_name='interactions')
    question = models.ForeignKey(Question, on_delete=models.CASCADE, related_name='interactions')
    response = models.TextField(blank=True)
    correct = models.BooleanField()
    score = models.FloatField(null=True, blank=True)
    time_taken = models.DurationField(help_text="Time taken to answer question")
    timestamp = models.DateTimeField()
    attempt_number = models.PositiveIntegerField(default=1)
    resource_viewed_before = models.BooleanField(default=False)
    
    def __str__(self):
        return f"{self.student} - {self.question} - {'Correct' if self.correct else 'Incorrect'}"
    
    class Meta:
        ordering = ['-timestamp']
        unique_together = ['student', 'question', 'attempt_number']


class KnowledgeState(models.Model):
    """Model representing a student's knowledge state for a topic."""
    student = models.ForeignKey(Student, on_delete=models.CASCADE, related_name='knowledge_states')
    topic = models.ForeignKey(Topic, on_delete=models.CASCADE, related_name='knowledge_states')
    proficiency_score = models.FloatField(
        help_text="Score between 0-1 representing mastery level"
    )
    confidence = models.FloatField(
        help_text="Confidence in the proficiency score prediction (0-1)"
    )
    last_updated = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return f"{self.student} - {self.topic} - {self.proficiency_score:.2f}"
    
    class Meta:
        ordering = ['-last_updated']
        unique_together = ['student', 'topic']
