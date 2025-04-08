from django.contrib import admin
from .models import DataUpload, ProcessingLog, DocumentUpload, ExtractedQuestion


class DataUploadProcessingLogInline(admin.TabularInline):
    model = ProcessingLog
    extra = 0
    readonly_fields = ('timestamp', 'level', 'message')
    can_delete = False
    fk_name = 'data_upload'
    
    def has_add_permission(self, request, obj=None):
        return False


class DocumentUploadProcessingLogInline(admin.TabularInline):
    model = ProcessingLog
    extra = 0
    readonly_fields = ('timestamp', 'level', 'message')
    can_delete = False
    fk_name = 'document_upload'
    
    def has_add_permission(self, request, obj=None):
        return False


@admin.register(DataUpload)
class DataUploadAdmin(admin.ModelAdmin):
    list_display = ('id', 'file', 'uploaded_by', 'uploaded_at', 'status', 'rows_processed', 'rows_failed')
    list_filter = ('status', 'uploaded_at')
    search_fields = ('file', 'uploaded_by__username')
    readonly_fields = ('uploaded_at', 'rows_processed', 'rows_failed')
    inlines = [DataUploadProcessingLogInline]


@admin.register(ProcessingLog)
class ProcessingLogAdmin(admin.ModelAdmin):
    list_display = ('data_upload', 'timestamp', 'level', 'message')
    list_filter = ('data_upload', 'level', 'timestamp')
    search_fields = ('message',)
    readonly_fields = ('timestamp',)


class ExtractedQuestionInline(admin.TabularInline):
    model = ExtractedQuestion
    extra = 0
    readonly_fields = ('extracted_at', 'question_number', 'page_number', 'is_processed', 
                      'topic', 'is_correct', 'confidence_score')
    fields = ('question_number', 'page_number', 'is_processed', 'topic', 
             'is_correct', 'confidence_score', 'question_text', 'student_answer', 'feedback')
    can_delete = False
    max_num = 0  # Don't allow adding new questions through admin
    
    def has_add_permission(self, request, obj=None):
        return False


@admin.register(DocumentUpload)
class DocumentUploadAdmin(admin.ModelAdmin):
    list_display = ('id', 'file', 'document_type', 'uploaded_by', 'uploaded_at', 
                   'status', 'questions_processed', 'questions_failed')
    list_filter = ('status', 'document_type', 'uploaded_at')
    search_fields = ('file', 'uploaded_by__username')
    readonly_fields = ('uploaded_at', 'questions_processed', 'questions_failed')
    inlines = [DocumentUploadProcessingLogInline, ExtractedQuestionInline]
    fieldsets = (
        (None, {
            'fields': ('file', 'document_type', 'uploaded_by', 'uploaded_at')
        }),
        ('Processing Status', {
            'fields': ('status', 'questions_processed', 'questions_failed', 'error_message')
        }),
        ('Associations', {
            'fields': ('student', 'course', 'assessment')
        }),
    )


from .utils import format_validation_metadata

@admin.register(ExtractedQuestion)
class ExtractedQuestionAdmin(admin.ModelAdmin):
    list_display = ('id', 'document', 'question_number', 'page_number', 'is_processed', 'has_detailed_validation')
    list_filter = ('is_processed', 'is_correct', 'document')
    search_fields = ('question_text', 'student_answer', 'feedback')
    readonly_fields = ('extracted_at', 'detailed_validation')
    
    fieldsets = (
        (None, {
            'fields': ('document', 'question_number', 'page_number', 'extracted_at')
        }),
        ('Question Content', {
            'fields': ('question_text', 'student_answer')
        }),
        ('Processing Results', {
            'fields': ('is_processed', 'topic', 'is_correct', 'confidence_score', 'feedback', 'detailed_validation')
        }),
    )
    
    def has_detailed_validation(self, obj):
        return bool(obj.validation_metadata)
    has_detailed_validation.boolean = True
    has_detailed_validation.short_description = 'Detailed Val'
    
    def detailed_validation(self, obj):
        return format_validation_metadata(obj.validation_metadata)
    detailed_validation.short_description = 'Validation Details'
