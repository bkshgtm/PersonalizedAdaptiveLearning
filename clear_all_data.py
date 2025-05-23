import os
import django
from django.db import transaction

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'pal_project.settings')
django.setup()

from django.contrib.auth.models import User
from core.models import Course, Topic, Resource, Assessment, Question, Student, StudentInteraction
from knowledge_graph.models import KnowledgeGraph, GraphEdge
from learning_paths.models import LearningPath, PathGenerationJob
from ml_models.models import KnowledgeTracingModel, TrainingJob, TopicMastery, PredictionBatch

def clear_all_data():
    with transaction.atomic():
        # Count all objects before deletion
        path_count = LearningPath.objects.count()
        job_count = PathGenerationJob.objects.count()
        student_count = Student.objects.count()
        interaction_count = StudentInteraction.objects.count()
        topic_count = Topic.objects.count()
        resource_count = Resource.objects.count()
        question_count = Question.objects.count()
        assessment_count = Assessment.objects.count()
        graph_count = KnowledgeGraph.objects.count()
        edge_count = GraphEdge.objects.count()
        model_count = KnowledgeTracingModel.objects.count()
        training_job_count = TrainingJob.objects.count()
        mastery_count = TopicMastery.objects.count()
        prediction_batch_count = PredictionBatch.objects.count()
        
        # Delete all objects
        print("Deleting all data...")
        
        # Delete learning paths and jobs
        LearningPath.objects.all().delete()
        PathGenerationJob.objects.all().delete()
        
        # Delete ML model related data
        TopicMastery.objects.all().delete()
        PredictionBatch.objects.all().delete()
        TrainingJob.objects.all().delete()
        KnowledgeTracingModel.objects.all().delete()
        
        # Delete student data
        StudentInteraction.objects.all().delete()
        Student.objects.all().delete()
        
        # Delete knowledge graph
        GraphEdge.objects.all().delete()
        KnowledgeGraph.objects.all().delete()
        
        # Delete course content
        Question.objects.all().delete()
        Assessment.objects.all().delete()
        Resource.objects.all().delete()
        Topic.objects.all().delete()
        Course.objects.all().delete()
        
        # Keep admin user
        
        print(f"Successfully deleted:")
        print(f"- {path_count} learning paths")
        print(f"- {job_count} generation jobs")
        print(f"- {student_count} students")
        print(f"- {interaction_count} student interactions")
        print(f"- {topic_count} topics")
        print(f"- {resource_count} resources")
        print(f"- {question_count} questions")
        print(f"- {assessment_count} assessments")
        print(f"- {graph_count} knowledge graphs")
        print(f"- {edge_count} graph edges")
        print(f"- {model_count} knowledge tracing models")
        print(f"- {training_job_count} training jobs")
        print(f"- {mastery_count} topic masteries")
        print(f"- {prediction_batch_count} prediction batches")

if __name__ == "__main__":
    clear_all_data()