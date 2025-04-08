from django.contrib import admin
from .models import (
    PathGenerator, PathGenerationJob, LearningPath, 
    LearningPathItem, LearningResource, PathCheckpoint
)


@admin.register(PathGenerator)
class PathGeneratorAdmin(admin.ModelAdmin):
    list_display = ('name', 'created_by', 'created_at', 'is_active')
    list_filter = ('is_active', 'created_at')
    search_fields = ('name', 'description')
    readonly_fields = ('created_at', 'updated_at')


class LearningPathInline(admin.TabularInline):
    model = LearningPath
    extra = 0
    readonly_fields = ('generated_at',)
    show_change_link = True


@admin.register(PathGenerationJob)
class PathGenerationJobAdmin(admin.ModelAdmin):
    list_display = (
        'id', 'generator', 'student', 'course', 'status', 
        'created_at', 'completed_at'
    )
    list_filter = ('status', 'created_at', 'course')
    search_fields = ('student__student_id', 'generator__name')
    readonly_fields = ('created_at', 'started_at', 'completed_at')
    inlines = [LearningPathInline]


class LearningPathItemInline(admin.TabularInline):
    model = LearningPathItem
    extra = 1
    readonly_fields = ('completed_at',)
    show_change_link = True


class PathCheckpointInline(admin.TabularInline):
    model = PathCheckpoint
    extra = 1
    readonly_fields = ('completed_at',)
    show_change_link = True


@admin.register(LearningPath)
class LearningPathAdmin(admin.ModelAdmin):
    list_display = (
        'id', 'name', 'student', 'course', 'status', 
        'generated_at', 'estimated_completion_time'
    )
    list_filter = ('status', 'generated_at', 'course')
    search_fields = ('name', 'student__student_id')
    readonly_fields = ('generated_at',)
    inlines = [LearningPathItemInline, PathCheckpointInline]


class LearningResourceInline(admin.TabularInline):
    model = LearningResource
    extra = 1
    readonly_fields = ('viewed_at',)


@admin.register(LearningPathItem)
class LearningPathItemAdmin(admin.ModelAdmin):
    list_display = (
        'id', 'path', 'topic', 'priority', 'status', 
        'proficiency_score', 'completed'
    )
    list_filter = ('status', 'completed', 'trend')
    search_fields = ('path__name', 'topic__name')
    readonly_fields = ('completed_at',)
    inlines = [LearningResourceInline]


@admin.register(LearningResource)
class LearningResourceAdmin(admin.ModelAdmin):
    list_display = ('id', 'path_item', 'resource', 'viewed')
    list_filter = ('viewed',)
    search_fields = ('path_item__topic__name', 'resource__title')
    readonly_fields = ('viewed_at',)


@admin.register(PathCheckpoint)
class PathCheckpointAdmin(admin.ModelAdmin):
    list_display = (
        'id', 'path', 'name', 'checkpoint_type', 'position', 'completed'
    )
    list_filter = ('checkpoint_type', 'completed')
    search_fields = ('name', 'path__name')
    readonly_fields = ('completed_at',)
    filter_horizontal = ('topics',)