from django.urls import path
from . import views

urlpatterns = [
    # Web views
    path('graphs/', views.graph_list, name='graph_list'),
    path('graphs/<int:graph_id>/', views.graph_detail, name='graph_detail'),
    path('graphs/create/', views.create_graph, name='create_graph'),
    path('graphs/upload/', views.upload_graph, name='upload_graph'),
    path('graphs/<int:graph_id>/set-active/', views.set_active_graph, name='set_active_graph'),
    path('graphs/<int:graph_id>/export/', views.export_graph, name='export_graph'),
    
    path('graphs/<int:graph_id>/topics/<int:topic_id>/', views.topic_detail, name='topic_detail'),
    path('graphs/<int:graph_id>/add-edge/', views.add_edge, name='add_edge'),
    path('edges/<int:edge_id>/remove/', views.remove_edge, name='remove_edge'),
    
    path('graphs/<int:graph_id>/learning-path/', views.learning_path, name='learning_path'),
    
    # API endpoints
    path('api/graph/active/', views.get_active_graph_api, name='get_active_graph_api'),
    path('api/graph/upload/', views.upload_graph_api, name='upload_graph_api'),
    path('api/topics/<int:topic_id>/prerequisites/', views.get_prerequisites_api, name='get_prerequisites_api'),
    path('api/topics/<int:topic_id>/next/', views.get_next_topics_api, name='get_next_topics_api'),
    path('api/learning-path/', views.find_learning_path_api, name='find_learning_path_api'),
    path('api/available-topics/', views.get_available_topics_api, name='get_available_topics_api'),
]