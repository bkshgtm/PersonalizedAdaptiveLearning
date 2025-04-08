from rest_framework import serializers
from core.models import Student, Course, Topic, Resource
from knowledge_graph.models import KnowledgeGraph
from ml_models.models import KnowledgeTracingModel, TopicMastery
from learning_paths.models import LearningPath, LearningPathItem, LearningResource

class StudentSerializer(serializers.ModelSerializer):
    """Serializer for the Student model."""
    
    class Meta:
        model = Student
        fields = [
            'student_id', 'academic_level', 'major', 'gpa', 
            'prior_knowledge_score', 'study_frequency', 'attendance_rate', 
            'participation_score'
        ]


class CourseSerializer(serializers.ModelSerializer):
    """Serializer for the Course model."""
    
    class Meta:
        model = Course
        fields = ['course_id', 'title', 'description']


class TopicSerializer(serializers.ModelSerializer):
    """Serializer for the Topic model."""
    
    class Meta:
        model = Topic
        fields = ['id', 'name', 'description', 'parent']


class ResourceSerializer(serializers.ModelSerializer):
    """Serializer for the Resource model."""
    
    class Meta:
        model = Resource
        fields = [
            'id', 'title', 'description', 'url', 'resource_type', 
            'difficulty', 'estimated_time'
        ]


class KnowledgeGraphSerializer(serializers.ModelSerializer):
    """Serializer for the KnowledgeGraph model."""
    
    class Meta:
        model = KnowledgeGraph
        fields = ['id', 'name', 'description', 'version', 'data', 'is_active']


class KnowledgeTracingModelSerializer(serializers.ModelSerializer):
    """Serializer for the KnowledgeTracingModel model."""
    
    class Meta:
        model = KnowledgeTracingModel
        fields = [
            'id', 'name', 'model_type', 'description', 'status',
            'hyperparameters', 'is_default'
        ]


class TopicMasterySerializer(serializers.ModelSerializer):
    """Serializer for the TopicMastery model."""
    topic_name = serializers.CharField(source='topic.name', read_only=True)
    
    class Meta:
        model = TopicMastery
        fields = [
            'id', 'topic', 'topic_name', 'mastery_score', 'confidence',
            'trend', 'predicted_at'
        ]


class LearningResourceSerializer(serializers.ModelSerializer):
    """Serializer for the LearningResource model."""
    resource_title = serializers.CharField(source='resource.title', read_only=True)
    resource_type = serializers.CharField(source='resource.resource_type', read_only=True)
    resource_url = serializers.URLField(source='resource.url', read_only=True)
    
    class Meta:
        model = LearningResource
        fields = [
            'id', 'resource', 'resource_title', 'resource_type', 'resource_url',
            'match_reason', 'viewed', 'viewed_at'
        ]


class LearningPathItemSerializer(serializers.ModelSerializer):
    """Serializer for the LearningPathItem model."""
    topic_name = serializers.CharField(source='topic.name', read_only=True)
    resources = LearningResourceSerializer(source='learningresource_set', many=True, read_only=True)
    
    class Meta:
        model = LearningPathItem
        fields = [
            'id', 'topic', 'topic_name', 'priority', 'status', 'proficiency_score',
            'trend', 'confidence_of_improvement', 'reason', 'estimated_review_time',
            'completed', 'completed_at', 'resources'
        ]


class LearningPathSerializer(serializers.ModelSerializer):
    """Serializer for the LearningPath model."""
    items = LearningPathItemSerializer(source='items', many=True, read_only=True)
    student_id = serializers.CharField(source='student.student_id', read_only=True)
    course_id = serializers.CharField(source='course.course_id', read_only=True)
    
    class Meta:
        model = LearningPath
        fields = [
            'id', 'name', 'description', 'student', 'student_id', 
            'course', 'course_id', 'generated_at', 'expires_at',
            'status', 'overall_progress', 'estimated_completion_time',
            'items'
        ]