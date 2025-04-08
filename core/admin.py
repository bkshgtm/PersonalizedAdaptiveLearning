from django.contrib import admin
from .models import (
    Student, Course, Topic, Resource, Assessment, Question, StudentInteraction
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


@admin.register(Topic)
class TopicAdmin(admin.ModelAdmin):
    list_display = ('name', 'parent', 'course')
    list_filter = ('course',)
    search_fields = ('name',)
    inlines = [SubtopicInline]


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