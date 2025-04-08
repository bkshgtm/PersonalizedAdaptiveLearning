from django.contrib import admin
from django import forms
from .models import KnowledgeGraph, GraphEdge
from core.models import Topic
import json
from pathlib import Path

class GraphEdgeInline(admin.TabularInline):
    model = GraphEdge
    extra = 1
    fk_name = 'graph'

class KnowledgeGraphAdminForm(forms.ModelForm):
    class Meta:
        model = KnowledgeGraph
        fields = '__all__'
    
    def clean_data(self):
        data = self.cleaned_data.get('data')
        if data:
            try:
                json.loads(data)
            except json.JSONDecodeError:
                raise forms.ValidationError("Invalid JSON data")
        return data

@admin.register(KnowledgeGraph)
class KnowledgeGraphAdmin(admin.ModelAdmin):
    form = KnowledgeGraphAdminForm
    list_display = ('name', 'version', 'is_active', 'created_at')
    list_filter = ('is_active',)
    search_fields = ('name', 'description')
    readonly_fields = ('created_at', 'updated_at')
    inlines = [GraphEdgeInline]
    actions = ['import_from_json']

    def import_from_json(self, request, queryset):
        from .services.graph_operations import GraphOperations
        for graph in queryset:
            try:
                if graph.data:
                    GraphOperations(graph.id).import_from_json(graph.data)
                    self.message_user(request, f"Successfully imported graph {graph.name}")
            except Exception as e:
                self.message_user(request, f"Error importing {graph.name}: {str(e)}", level='error')
    import_from_json.short_description = "Import graph from JSON data"

@admin.register(GraphEdge)
class GraphEdgeAdmin(admin.ModelAdmin):
    list_display = ('source_topic', 'target_topic', 'relationship_type', 'weight')
    list_filter = ('relationship_type', 'graph')
    search_fields = ('source_topic__name', 'target_topic__name')
