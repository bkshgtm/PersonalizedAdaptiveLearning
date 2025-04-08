from django.contrib import admin
from .models import KnowledgeTracingModel, TrainingJob, PredictionBatch, TopicMastery


@admin.register(KnowledgeTracingModel)
class KnowledgeTracingModelAdmin(admin.ModelAdmin):
    list_display = ('name', 'model_type', 'course', 'status', 'created_at', 'is_default')
    list_filter = ('model_type', 'status', 'course', 'is_default')
    search_fields = ('name', 'description')
    readonly_fields = ('created_at', 'updated_at')


class TopicMasteryInline(admin.TabularInline):
    model = TopicMastery
    extra = 0
    readonly_fields = ('student', 'topic', 'mastery_score', 'confidence', 'predicted_at', 'trend')
    can_delete = False
    
    def has_add_permission(self, request, obj=None):
        return False


@admin.register(TrainingJob)
class TrainingJobAdmin(admin.ModelAdmin):
    list_display = (
        'id', 'model', 'status', 'created_at', 'total_students',
        'total_interactions'
    )
    list_filter = ('status', 'model__model_type', 'created_at')
    readonly_fields = ('created_at', 'started_at', 'completed_at')
    

@admin.register(PredictionBatch)
class PredictionBatchAdmin(admin.ModelAdmin):
    list_display = (
        'id', 'model', 'status', 'created_at', 'total_students',
        'processed_students'
    )
    list_filter = ('status', 'model__model_type', 'created_at')
    readonly_fields = ('created_at', 'started_at', 'completed_at')
    inlines = [TopicMasteryInline]


@admin.register(TopicMastery)
class TopicMasteryAdmin(admin.ModelAdmin):
    list_display = (
        'student', 'topic', 'mastery_score', 'confidence',
        'trend', 'predicted_at'
    )
    list_filter = ('prediction_batch', 'trend', 'topic', 'predicted_at')
    search_fields = ('student__student_id', 'topic__name')
    readonly_fields = ('predicted_at',)