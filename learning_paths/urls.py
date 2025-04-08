from django.urls import path
from . import views

urlpatterns = [
    # Web views
    path('generators/', views.generator_list, name='generator_list'),
    path('generators/<int:generator_id>/', views.generator_detail, name='generator_detail'),
    path('generators/create/', views.create_generator, name='create_generator'),
    path('generators/<int:generator_id>/set-active/', views.set_active_generator, name='set_active_generator'),
    
    path('jobs/', views.job_list, name='job_list'),
    path('jobs/<int:job_id>/', views.job_detail, name='job_detail'),
    path('generate-path/', views.generate_path, name='generate_path'),
    path('refresh-course-paths/<str:course_id>/', views.refresh_course_paths, name='refresh_course_paths'),
    
    path('paths/', views.path_list, name='path_list'),
    path('paths/<int:path_id>/', views.path_detail, name='path_detail'),
    path('paths/<int:path_id>/archive/', views.archive_path, name='archive_path'),
    
    path('students/<str:student_id>/paths/', views.student_paths, name='student_paths'),
    
    path('resources/<int:resource_id>/mark-viewed/', views.mark_resource_viewed, name='mark_resource_viewed'),
    path('items/<int:item_id>/mark-completed/', views.mark_item_completed, name='mark_item_completed'),
    path('checkpoints/<int:checkpoint_id>/mark-completed/', views.mark_checkpoint_completed, name='mark_checkpoint_completed'),
    
    # API endpoints
    path('api/student/<str:student_id>/path/', views.get_student_path_api, name='get_student_path_api'),
    path('api/generate-path/', views.generate_path_api, name='generate_path_api'),
    path('api/resources/<int:resource_id>/update-status/', views.update_resource_status_api, name='update_resource_status_api'),
    path('api/items/<int:item_id>/update-status/', views.update_item_status_api, name='update_item_status_api'),
    path('api/checkpoints/<int:checkpoint_id>/update-status/', views.update_checkpoint_status_api, name='update_checkpoint_status_api'),
]