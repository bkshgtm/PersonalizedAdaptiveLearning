from django.urls import path
from . import views

app_name = 'learning_paths'

urlpatterns = [
    path('', views.student_list, name='student_list'),
    path('student/<str:student_id>/', views.student_paths, name='student_paths'),
    path('view/<int:path_id>/', views.path_detail, name='path_detail'),
    path('topic-resources/<int:topic_id>/', views.topic_resources, name='topic_resources'),
    path('dashboard/', views.learning_path_dashboard, name='dashboard'),
    path('generate/<str:student_id>/', views.generate_learning_path, name='generate_path'),
    path('test-models/', views.test_integrated_models, name='test_models'),
]
