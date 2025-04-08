from django.urls import path
from . import views

urlpatterns = [
    # Web views
    path('models/', views.model_list, name='model_list'),
    path('models/<int:model_id>/', views.model_detail, name='model_detail'),
    path('models/create/', views.create_model, name='create_model'),
    path('models/<int:model_id>/set-default/', views.set_default_model, name='set_default_model'),
    path('models/<int:model_id>/train/', views.train_model, name='train_model'),
    path('models/<int:model_id>/predict/', views.generate_predictions, name='generate_predictions'),
    
    path('jobs/', views.job_list, name='job_list'),
    path('jobs/<int:job_id>/', views.job_detail, name='job_detail'),
    
    path('batches/', views.batch_list, name='batch_list'),
    path('batches/<int:batch_id>/', views.batch_detail, name='batch_detail'),
    
    path('students/<str:student_id>/masteries/', views.student_masteries, name='student_masteries'),
    path('topics/<int:topic_id>/masteries/', views.topic_masteries, name='topic_masteries'),
    
    # API endpoints
    path('api/models/', views.get_models_api, name='get_models_api'),
    path('api/models/create/', views.create_model_api, name='create_model_api'),
    path('api/models/<int:model_id>/train/', views.train_model_api, name='train_model_api'),
    path('api/models/<int:model_id>/predict/', views.generate_predictions_api, name='generate_predictions_api'),
    path('api/jobs/<int:job_id>/status/', views.get_job_status_api, name='get_job_status_api'),
    path('api/students/<str:student_id>/masteries/', views.get_student_masteries_api, name='get_student_masteries_api'),
]