from django.urls import path
from .views import (
    ModelInfoView, LearningPathsView,
    TrainModelView, TrainStreamView
)

app_name = 'ml_models'

urlpatterns = [
    path('model-info/', ModelInfoView.as_view(), name='model_info'),
    path('learning-paths/', LearningPathsView.as_view(), name='learning_paths'),
]
