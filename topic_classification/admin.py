from django.contrib import admin
from .models import ClassificationModel, ClassificationJob, ClassificationResult


@admin.register(ClassificationModel)
class ClassificationModelAdmin(admin.ModelAdmin):
    list_display = ('name', 'model_type', 'status', 'created_at', 'is_default')
    list_filter = ('model_type', 'status', 'is_default')
    search_fields = ('name', 'description')
    readonly_fields = ('created_at', 'updated_at')


class ClassificationResultInline(admin.TabularInline):
    model = ClassificationResult
    extra = 0
    readonly_fields = ('question', 'topic', 'confidence', 'classified_at')
    can_delete = False
    
    def has_add_permission(self, request, obj=None):
        return False


@admin.register(ClassificationJob)
class ClassificationJobAdmin(admin.ModelAdmin):
    list_display = (
        'id', 'model', 'status', 'created_at', 'total_questions',
        'classified_questions', 'failed_questions'
    )
    list_filter = ('status', 'model', 'created_at')
    readonly_fields = ('created_at', 'started_at', 'completed_at')
    inlines = [ClassificationResultInline]


@admin.register(ClassificationResult)
class ClassificationResultAdmin(admin.ModelAdmin):
    list_display = (
        'id', 'job', 'question', 'topic', 'confidence',
        'classified_at', 'is_verified'
    )
    list_filter = ('job', 'topic', 'is_verified', 'classified_at')
    search_fields = ('question__question_id', 'question__text')
    readonly_fields = ('classified_at',)