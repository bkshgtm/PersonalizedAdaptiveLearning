from django.urls import path
from . import views

app_name = 'data_ingestion'

urlpatterns = [
    # CSV Upload Web views
    path('upload/', views.upload_data, name='upload_data'),
    path('uploads/', views.upload_list, name='upload_list'),
    path('uploads/<int:upload_id>/', views.upload_detail, name='upload_detail'),
    path('uploads/<int:upload_id>/status/', views.check_upload_status, name='check_upload_status'),
    path('uploads/<int:upload_id>/error-report/', views.download_error_report, name='download_error_report'),
    
    # Document Upload Web views
    path('upload-document/', views.upload_document, name='upload_document'),
    path('documents/', views.document_list, name='document_list'),
    path('documents/<int:document_id>/', views.document_detail, name='document_detail'),
    path('documents/<int:document_id>/status/', views.check_document_status, name='check_document_status'),
    
    # API endpoints for CSV uploads
    path('api/upload/', views.upload_data_api, name='upload_data_api'),
    path('api/uploads/<int:upload_id>/status/', views.upload_status_api, name='upload_status_api'),
    
    # API endpoints for Document uploads
    path('api/upload-document/', views.upload_document_api, name='upload_document_api'),
    path('api/documents/<int:document_id>/status/', views.document_status_api, name='document_status_api'),
    path('api/questions/<int:question_id>/', views.extracted_question_api, name='extracted_question_api'),
]
