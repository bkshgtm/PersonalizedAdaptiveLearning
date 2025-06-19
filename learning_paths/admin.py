from django.contrib import admin
from .models import (
    LearningPath, WeakTopic, RecommendedTopic, TopicResource,
    LearningPathProgress, StudySession, PathFeedback
)


class WeakTopicInline(admin.TabularInline):
    model = WeakTopic
    extra = 0
    readonly_fields = ('current_mastery',)
    fields = ('topic', 'current_mastery', 'order')


class RecommendedTopicInline(admin.TabularInline):
    model = RecommendedTopic
    extra = 0
    readonly_fields = ('confidence', 'estimated_time_hours')
    fields = ('topic', 'confidence', 'recommended_difficulty', 'priority', 'completed')


class TopicResourceInline(admin.TabularInline):
    model = TopicResource
    extra = 0
    readonly_fields = ('viewed_at', 'completed_at')
    fields = ('title', 'resource_type', 'difficulty', 'estimated_time', 'viewed', 'completed', 'order')


@admin.register(LearningPath)
class LearningPathAdmin(admin.ModelAdmin):
    list_display = (
        'id', 'student', 'course', 'name', 'status', 
        'generated_at', 'total_estimated_time', 'overall_progress'
    )
    list_filter = ('status', 'generated_at', 'course')
    search_fields = ('name', 'student__student_id', 'student__first_name', 'student__last_name')
    readonly_fields = ('generated_at', 'weak_topics_count', 'recommended_topics_count')
    inlines = [WeakTopicInline, RecommendedTopicInline]
    
    fieldsets = (
        ('Basic Information', {
            'fields': ('student', 'course', 'name', 'description', 'status')
        }),
        ('Path Statistics', {
            'fields': ('total_estimated_time', 'weak_topics_count', 'recommended_topics_count', 'overall_progress')
        }),
        ('Student Profile', {
            'fields': ('student_stats',),
            'classes': ('collapse',)
        }),
        ('Timestamps', {
            'fields': ('generated_at', 'expires_at', 'last_accessed'),
            'classes': ('collapse',)
        }),
    )


@admin.register(WeakTopic)
class WeakTopicAdmin(admin.ModelAdmin):
    list_display = ('learning_path', 'topic', 'current_mastery', 'order')
    list_filter = ('learning_path__course', 'topic')
    search_fields = ('learning_path__student__student_id', 'topic__name')
    readonly_fields = ('current_mastery',)
    inlines = [TopicResourceInline]
    
    def get_queryset(self, request):
        return super().get_queryset(request).select_related('learning_path__student', 'topic')


@admin.register(RecommendedTopic)
class RecommendedTopicAdmin(admin.ModelAdmin):
    list_display = (
        'learning_path', 'topic', 'confidence', 'recommended_difficulty', 
        'priority', 'completed', 'should_study_prerequisites_first'
    )
    list_filter = (
        'learning_path__course', 'recommended_difficulty', 'completed', 
        'should_study_prerequisites_first'
    )
    search_fields = ('learning_path__student__student_id', 'topic__name')
    readonly_fields = ('confidence', 'estimated_time_hours', 'started_at', 'completed_at')
    inlines = [TopicResourceInline]
    
    fieldsets = (
        ('Basic Information', {
            'fields': ('learning_path', 'topic', 'priority')
        }),
        ('Recommendation Details', {
            'fields': ('confidence', 'recommended_difficulty', 'estimated_time_hours')
        }),
        ('Prerequisites', {
            'fields': ('prerequisites', 'unmet_prerequisites', 'should_study_prerequisites_first'),
            'classes': ('collapse',)
        }),
        ('Progress Tracking', {
            'fields': ('completed', 'started_at', 'completed_at', 'progress_percentage')
        }),
        ('Relationships', {
            'fields': ('related_topics',),
            'classes': ('collapse',)
        }),
    )
    
    def get_queryset(self, request):
        return super().get_queryset(request).select_related('learning_path__student', 'topic')


@admin.register(TopicResource)
class TopicResourceAdmin(admin.ModelAdmin):
    list_display = (
        'title', 'get_topic_name', 'resource_type', 'difficulty', 
        'estimated_time', 'viewed', 'completed', 'rating'
    )
    list_filter = ('resource_type', 'difficulty', 'viewed', 'completed', 'rating')
    search_fields = ('title', 'description', 'url')
    readonly_fields = ('viewed_at', 'completed_at')
    
    fieldsets = (
        ('Resource Information', {
            'fields': ('weak_topic', 'recommended_topic', 'title', 'description', 'url')
        }),
        ('Resource Properties', {
            'fields': ('resource_type', 'difficulty', 'estimated_time', 'order')
        }),
        ('Progress Tracking', {
            'fields': ('viewed', 'viewed_at', 'completed', 'completed_at', 'rating')
        }),
        ('Student Notes', {
            'fields': ('notes',),
            'classes': ('collapse',)
        }),
    )
    
    def get_topic_name(self, obj):
        if obj.weak_topic:
            return f"Weak: {obj.weak_topic.topic.name}"
        elif obj.recommended_topic:
            return f"Rec: {obj.recommended_topic.topic.name}"
        return "No topic"
    get_topic_name.short_description = "Topic"
    
    def get_queryset(self, request):
        return super().get_queryset(request).select_related(
            'weak_topic__topic', 'recommended_topic__topic'
        )


@admin.register(LearningPathProgress)
class LearningPathProgressAdmin(admin.ModelAdmin):
    list_display = (
        'learning_path', 'topics_completed', 'resources_completed', 
        'study_streak_days', 'completion_rate', 'last_study_session'
    )
    list_filter = ('last_study_session', 'study_streak_days')
    search_fields = ('learning_path__student__student_id', 'learning_path__name')
    readonly_fields = ('total_time_spent', 'average_session_time')
    
    fieldsets = (
        ('Progress Overview', {
            'fields': ('learning_path', 'topics_started', 'topics_completed', 'resources_viewed', 'resources_completed')
        }),
        ('Time Tracking', {
            'fields': ('total_time_spent', 'last_study_session', 'study_streak_days', 'average_session_time')
        }),
        ('Performance Metrics', {
            'fields': ('completion_rate',)
        }),
        ('Milestones', {
            'fields': ('milestones_achieved', 'next_milestone'),
            'classes': ('collapse',)
        }),
    )


@admin.register(StudySession)
class StudySessionAdmin(admin.ModelAdmin):
    list_display = (
        'learning_path', 'started_at', 'duration', 'topics_completed', 
        'resources_completed', 'effectiveness_rating'
    )
    list_filter = ('started_at', 'effectiveness_rating', 'difficulty_rating')
    search_fields = ('learning_path__student__student_id', 'notes')
    readonly_fields = ('duration',)
    filter_horizontal = ('topics_studied',)
    
    fieldsets = (
        ('Session Information', {
            'fields': ('learning_path', 'started_at', 'ended_at', 'duration')
        }),
        ('Study Content', {
            'fields': ('topics_studied', 'resources_accessed')
        }),
        ('Session Outcomes', {
            'fields': ('topics_completed', 'resources_completed', 'notes')
        }),
        ('Session Quality', {
            'fields': ('effectiveness_rating', 'difficulty_rating')
        }),
    )


@admin.register(PathFeedback)
class PathFeedbackAdmin(admin.ModelAdmin):
    list_display = (
        'learning_path', 'feedback_type', 'rating', 'topic', 
        'resource', 'created_at'
    )
    list_filter = ('feedback_type', 'rating', 'created_at')
    search_fields = ('learning_path__student__student_id', 'comment')
    readonly_fields = ('created_at',)
    
    fieldsets = (
        ('Feedback Information', {
            'fields': ('learning_path', 'feedback_type', 'rating')
        }),
        ('Specific Context', {
            'fields': ('topic', 'resource')
        }),
        ('Feedback Content', {
            'fields': ('comment',)
        }),
        ('Metadata', {
            'fields': ('created_at',)
        }),
    )
