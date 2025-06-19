from django.shortcuts import render
from django.views import View
from django.http import JsonResponse, StreamingHttpResponse
from django.core.management import call_command
from django.conf import settings
import torch
import subprocess
import json
import time
from ml_models.models import KnowledgeTracingModel, TopicMastery
from ml_models.ml.dkt import DKTModel
from ml_models.ml.sakt import SAKTModel
from core.models import Student, Topic, Resource
from learning_paths.models import LearningPath
# from learning_paths.services.path_generator import LearningPathGenerator as PathGenerator
from learning_paths.ml.adaptive_path_lstm import DjangoIntegratedPathGenerator as PathGenerator
import json
import os

class ModelInfoView(View):
    template_name = 'ml_models/model_info.html'

    def get(self, request, *args, **kwargs):
        # Get active model
        try:
            model = KnowledgeTracingModel.objects.filter(status='active').latest('updated_at')
        except KnowledgeTracingModel.DoesNotExist:
            model = None
        
        # Get latest topic mastery for all students
        topic_masteries = TopicMastery.objects.filter(
            prediction_batch__status='completed'
        ).order_by('topic__name', '-predicted_at').distinct('topic__name')
        
        context = {
            'model': model,
            'topic_masteries': topic_masteries
        }
        return render(request, self.template_name, context)

class PredictView(View):
    template_name = 'ml_models/prediction_results.html'

    def get(self, request, *args, **kwargs):
        num_topics = Topic.objects.count()
        model = DKTModel(num_topics=num_topics)  # Default to DKT model
        students = Student.objects.all()
        topics = Topic.objects.all()
        
        results = {}
        for student in students:
            predictions = model.predict(student)
            results[student.student_id] = predictions
        
        context = {
            'results': results,
            'topics': topics
        }
        return render(request, self.template_name, context)

class TrainModelView(View):
    template_name = 'ml_models/train_model.html'

    def get(self, request, *args, **kwargs):
        return render(request, self.template_name)

class TrainStreamView(View):
    def get(self, request, *args, **kwargs):
        model_type = request.GET.get('model', 'dkt')
        epochs = request.GET.get('epochs', 10)
        batch_size = request.GET.get('batch_size', 32)
        
        def event_stream():
            proc = subprocess.Popen(
                ['python', 'manage.py', 'train_models',
                 '--model', model_type,
                 '--epochs', str(epochs),
                 '--batch_size', str(batch_size)],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True
            )
            
            for line in proc.stdout:
                if 'Epoch' in line:
                    progress = int(line.split('/')[0].split()[-1]) / int(epochs) * 100
                    yield f"data: {json.dumps({'progress': progress, 'message': line.strip()})}\n\n"
                elif 'loss' in line.lower():
                    yield f"data: {json.dumps({'message': line.strip()})}\n\n"
            
            # Set session flag when training completes
            request.session['training_complete'] = True
            yield f"data: {json.dumps({'complete': True, 'message': 'Training completed', 'redirect': '/ml-models/learning-paths/'})}\n\n"
            proc.wait()
        
        return StreamingHttpResponse(event_stream(), content_type='text/event-stream')

class PredictProgressView(View):
    template_name = 'ml_models/predict_progress.html'

    def get(self, request, *args, **kwargs):
        return render(request, self.template_name)

class PredictStreamView(View):
    def get(self, request, *args, **kwargs):
        def event_stream():
            try:
                # Initialize model with required parameters
                model = DKTModel(
                    num_topics=Topic.objects.count(),
                    hidden_size=64,
                    num_layers=2
                )
                
                # Load trained model weights and metadata
                model_dir = os.path.join(settings.BASE_DIR, 'trained_models')
                model_path = os.path.join(model_dir, 'latest_model.pt')
                metadata_path = os.path.join(model_dir, 'metadata.json')
                
                if not os.path.exists(model_dir):
                    os.makedirs(model_dir)
                
                #  error handling
                try:
                    if os.path.exists(model_path):
                        model.load(model_path)
                        if os.path.exists(metadata_path):
                            try:
                                with open(metadata_path) as f:
                                    metadata = json.load(f)
                                    model.set_metadata(metadata)
                            except (json.JSONDecodeError, IOError) as e:
                                yield f"data: {json.dumps({'warning': True, 'message': f'Metadata load failed: {str(e)}'})}\n\n"
                    else:
                        yield f"data: {json.dumps({'warning': True, 'message': 'No trained model found, using fresh model'})}\n\n"
                except Exception as e:
                    yield f"data: {json.dumps({'error': True, 'message': f'Model loading failed: {str(e)}'})}\n\n"
                    return
                
                students = list(Student.objects.all())
                total = len(students)
                
                for i, student in enumerate(students):
                    progress = (i + 1) / total * 100
                    predictions = model.predict(student)
                    yield f"data: {json.dumps({'progress': progress, 'message': f'Processed student {student.student_id}'})}\n\n"
                
                yield f"data: {json.dumps({'complete': True, 'message': 'All predictions completed', 'redirect': '/ml-models/learning-paths/'})}\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'error': True, 'message': f'Prediction failed: {str(e)}'})}\n\n"
        
        response = StreamingHttpResponse(event_stream(), content_type='text/event-stream')
        response['Cache-Control'] = 'no-cache'
        return response

class LearningPathsView(View):
    template_name = 'ml_models/learning_paths.html'

    def get(self, request, *args, **kwargs):
        # Clear training complete flag
        if 'training_complete' in request.session:
            del request.session['training_complete']
        
        # Get all existing learning paths from database
        paths = {}
        for path in LearningPath.objects.all().prefetch_related('items__topic', 'items__resources'):
            student_id = path.student.student_id
            if student_id not in paths:
                paths[student_id] = []
            
            path_data = {
                'id': path.id,
                'name': path.name,
                'created': path.generated_at,
                'items': [],
                'avg_proficiency': 0  # Initialize before calculation
            }
            
            for item in path.items.all():
                item_data = {
                    'topic': item.topic.name,
                    'status': item.status,
                    'proficiency': item.proficiency_score,
                    'resources': [{
                        'title': res.resource.title,
                        'type': res.resource.resource_type,
                        'url': res.resource.url
                    } for res in item.resources.all()]
                }
                path_data['items'].append(item_data)
            
            # Calculate average proficiency
            if path_data['items']:
                total_proficiency = sum(item['proficiency'] for item in path_data['items'])
                path_data['avg_proficiency'] = total_proficiency / len(path_data['items'])
            
            paths[student_id].append(path_data)
        
        context = {
            'paths': paths
        }
        return render(request, self.template_name, context)
