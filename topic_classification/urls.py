from django.urls import path
from . import views

urlpatterns = [
    # Web views
    path('models/', views.model_list, name='model_list'),
    path('models/<int:model_id>/', views.model_detail, name='model_detail'),
    path('models/create/', views.create_model, name='create_model'),
    path('models/<int:model_id>/set-default/', views.set_default_model, name='set_default_model'),
    path('models/<int:model_id>/train/', views.train_model, name='train_model'),
    path('models/<int:model_id>/classify/', views.start_classification, name='start_classification'),
    
    path('jobs/', views.job_list, name='job_list'),
    path('jobs/<int:job_id>/', views.job_detail, name='job_detail'),
    
    path('results/<int:result_id>/verify/', views.verify_classification, name='verify_classification'),
    path('results/<int:result_id>/update/', views.update_classification, name='update_classification'),
    
    # API endpoints
    path('api/models/', views.create_model_api, name='create_model_api'),
    path('api/models/<int:model_id>/train/', views.train_model_api, name='train_model_api'),
    path('api/models/<int:model_id>/classify/', views.start_classification_api, name='start_classification_api'),
]