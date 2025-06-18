from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from django.views.generic import RedirectView

urlpatterns = [
    # Admin interface
    path('admin/', admin.site.urls),
    
    # Authentication URLs
    path('accounts/', include('django.contrib.auth.urls')),
    
    # Home page (public)
    path('', include('core.urls')),
    
    # API endpoints
    path('api/', include('api.urls')),
    
    # App-specific URLs
    path('data-ingestion/', include(('data_ingestion.urls', 'data_ingestion'), namespace='data_ingestion')),
    path('topic-classification/', include(('topic_classification.urls', 'topic_classification'), namespace='topic_classification')),
    path('knowledge-graph/', include(('knowledge_graph.urls', 'knowledge_graph'), namespace='knowledge_graph')),
    path('ml-models/', include('ml_models.urls')),
    path('learning-paths/', include('learning_paths.urls')),
]

# Serve static and media files in development
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
    
    # Django Debug Toolbar (optional)
    try:
        import debug_toolbar
        urlpatterns.append(path('__debug__/', include(debug_toolbar.urls)))
    except ImportError:
        pass
