from django.contrib import admin
from knowledge_graph.models import GraphEdge
from .models import (
    Student, Course, Topic, Resource, Assessment, Question, StudentInteraction, KnowledgeState
)

@admin.register(Student)
class StudentAdmin(admin.ModelAdmin):
    list_display = ('student_id', 'academic_level', 'major', 'gpa', 'study_frequency')
    list_filter = ('academic_level', 'study_frequency')
    search_fields = ('student_id', 'major')


@admin.register(Course)
class CourseAdmin(admin.ModelAdmin):
    list_display = ('course_id', 'title')
    search_fields = ('course_id', 'title')
    filter_horizontal = ('students',)


class SubtopicInline(admin.TabularInline):
    model = Topic
    fk_name = 'parent'
    extra = 3


class OutgoingEdgeInline(admin.TabularInline):
    model = GraphEdge
    fk_name = 'source_topic'
    extra = 0
    verbose_name = 'Prerequisite For'
    verbose_name_plural = 'Prerequisites For Other Topics'
    readonly_fields = ('relationship_type', 'weight')
    can_delete = False

class IncomingEdgeInline(admin.TabularInline):
    model = GraphEdge
    fk_name = 'target_topic'
    extra = 0
    verbose_name = 'Requires These Prerequisites'
    verbose_name_plural = 'Required Prerequisites'
    readonly_fields = ('relationship_type', 'weight')
    can_delete = False

@admin.register(Topic)
class TopicAdmin(admin.ModelAdmin):
    list_display = ('name', 'parent', 'course', 'has_prerequisites', 'is_prerequisite_for')
    list_filter = ('course', 'incoming_edges__relationship_type')
    search_fields = ('name',)
    inlines = [SubtopicInline, OutgoingEdgeInline, IncomingEdgeInline]
    
    def has_prerequisites(self, obj):
        return obj.incoming_edges.exists()
    has_prerequisites.boolean = True
    has_prerequisites.short_description = 'Has Prereqs'
    
    def is_prerequisite_for(self, obj):
        return obj.outgoing_edges.exists()
    is_prerequisite_for.boolean = True
    is_prerequisite_for.short_description = 'Is Prereq For'


@admin.register(Resource)
class ResourceAdmin(admin.ModelAdmin):
    list_display = ('title', 'resource_type', 'difficulty', 'estimated_time')
    list_filter = ('resource_type', 'difficulty')
    search_fields = ('title', 'description')
    filter_horizontal = ('topics',)


@admin.register(Assessment)
class AssessmentAdmin(admin.ModelAdmin):
    list_display = ('assessment_id', 'title', 'assessment_type', 'course', 'date', 'proctored')
    list_filter = ('assessment_type', 'course', 'proctored')
    search_fields = ('assessment_id', 'title')


class StudentInteractionInline(admin.TabularInline):
    model = StudentInteraction
    extra = 0
    readonly_fields = ('timestamp',)


@admin.register(Question)
class QuestionAdmin(admin.ModelAdmin):
    list_display = ('question_id', 'assessment', 'question_type', 'topic')
    list_filter = ('assessment', 'question_type', 'topic')
    search_fields = ('question_id', 'text')
    inlines = [StudentInteractionInline]


@admin.register(StudentInteraction)
class StudentInteractionAdmin(admin.ModelAdmin):
    list_display = ('student', 'question', 'correct', 'score', 'attempt_number', 'timestamp')
    list_filter = ('correct', 'question__assessment', 'timestamp')
    search_fields = ('student__student_id', 'question__question_id')


@admin.register(KnowledgeState)
class KnowledgeStateAdmin(admin.ModelAdmin):
    list_display = ('student', 'topic', 'proficiency_score', 'confidence', 'last_updated')
    list_filter = ('topic', 'last_updated')
    search_fields = ('student__student_id', 'topic__name')
    readonly_fields = ('last_updated',)
    
    def get_queryset(self, request):
        return super().get_queryset(request).select_related('student', 'topic')
