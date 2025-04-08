from django.urls import path
from . import views

urlpatterns = [
    # Home page (public)
    path('', views.home, name='home'),
    
    # Dashboard (requires login)
    path('dashboard/', views.dashboard, name='dashboard'),
    
    # Student views
    path('students/', views.student_list, name='student_list'),
    path('students/<str:student_id>/', views.student_detail, name='student_detail'),
    path('students/create/', views.create_student, name='create_student'),
    path('students/<str:student_id>/edit/', views.edit_student, name='edit_student'),
    
    # Course views
    path('courses/', views.course_list, name='course_list'),
    path('courses/<str:course_id>/', views.course_detail, name='course_detail'),
    path('courses/create/', views.create_course, name='create_course'),
    path('courses/<str:course_id>/edit/', views.edit_course, name='edit_course'),
    path('courses/<str:course_id>/students/', views.course_students, name='course_students'),
    
    # Topic views
    path('topics/', views.topic_list, name='topic_list'),
    path('topics/<int:topic_id>/', views.topic_detail, name='topic_detail'),
    path('topics/create/', views.create_topic, name='create_topic'),
    path('topics/<int:topic_id>/edit/', views.edit_topic, name='edit_topic'),
    
    # Resource views
    path('resources/', views.resource_list, name='resource_list'),
    path('resources/<int:resource_id>/', views.resource_detail, name='resource_detail'),
    path('resources/create/', views.create_resource, name='create_resource'),
    path('resources/<int:resource_id>/edit/', views.edit_resource, name='edit_resource'),
    
    # Assessment views
    path('assessments/', views.assessment_list, name='assessment_list'),
    path('assessments/<str:assessment_id>/', views.assessment_detail, name='assessment_detail'),
    path('assessments/create/', views.create_assessment, name='create_assessment'),
    path('assessments/<str:assessment_id>/edit/', views.edit_assessment, name='edit_assessment'),
    
    # Question views
    path('questions/', views.question_list, name='question_list'),
    path('questions/<str:question_id>/', views.question_detail, name='question_detail'),
    path('questions/create/', views.create_question, name='create_question'),
    path('questions/<str:question_id>/edit/', views.edit_question, name='edit_question'),
    
    # Interaction views
    path('interactions/', views.interaction_list, name='interaction_list'),
    path('interactions/create/', views.create_interaction, name='create_interaction'),
    
    # API endpoints
    path('api/students/', views.get_students_api, name='get_students_api'),
    path('api/students/<str:student_id>/', views.get_student_api, name='get_student_api'),
    path('api/students/<str:student_id>/performance/', views.get_student_performance_api, name='get_student_performance_api'),
    
    path('api/courses/', views.get_courses_api, name='get_courses_api'),
    path('api/courses/<str:course_id>/', views.get_course_api, name='get_course_api'),
    
    path('api/topics/', views.get_topics_api, name='get_topics_api'),
    path('api/topics/<int:topic_id>/', views.get_topic_api, name='get_topic_api'),
    
    path('api/resources/', views.get_resources_api, name='get_resources_api'),
    path('api/resources/<int:resource_id>/', views.get_resource_api, name='get_resource_api'),
]
