from django.urls import path, include
from rest_framework.authtoken.views import obtain_auth_token
from . import views

urlpatterns = [
    # Authentication
    path('auth/token/', obtain_auth_token, name='api_token_auth'),
    
    # Integrated API endpoints
    path('students/<str:student_id>/profile/', views.student_profile, name='student_profile_api'),
    path('students/<str:student_id>/recommendations/', views.generate_recommendations, name='generate_recommendations_api'),
    path('recommendation-status/', views.check_recommendation_status, name='check_recommendation_status_api'),
    path('monitoring/dashboard/', views.monitoring_dashboard, name='monitoring_dashboard_api'),
    path('courses/<str:course_id>/recommendations/', views.recommendation_dashboard, name='recommendation_dashboard_api'),
    
    # Data ingestion
    path('data-ingestion/', include('data_ingestion.urls')),
    
    # Topic classification
    path('topic-classification/', include('topic_classification.urls')),
    
    # Answer validation endpoint
    path('validate-answer/', views.validate_answer_api, name='validate_answer_api'),
    
    # Knowledge graph
    path('knowledge-graph/', include('knowledge_graph.urls')),
    
    # ML models
    path('ml-models/', include('ml_models.urls')),
    
    # Learning paths
    path('learning-paths/', include('learning_paths.urls')),
    
    # Documentation
    path('docs/', views.api_documentation, name='api_documentation'),
]