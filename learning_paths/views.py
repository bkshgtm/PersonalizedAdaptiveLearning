from django.shortcuts import render, get_object_or_404, redirect
from django.http import JsonResponse
from django.contrib import messages
from django.db.models import Count, Avg, Max
from django.db import models
from django.utils import timezone
from datetime import timedelta
import json

from core.models import Student, Course
from .models import LearningPath, WeakTopic, RecommendedTopic, TopicResource
from .ml.adaptive_path_lstm import DjangoIntegratedPathGenerator


def student_list(request):
    """
    Level 1: Display all students with learning paths in a beautiful table.
    """
    # Get students who have learning paths
    students_with_paths = Student.objects.annotate(
        path_count=Count('learning_paths'),
        latest_path_date=Max('learning_paths__generated_at')
    ).filter(path_count__gt=0).order_by('-latest_path_date')
    
    # Also get students without paths for completeness
    students_without_paths = Student.objects.annotate(
        path_count=Count('learning_paths')
    ).filter(path_count=0)
    
    context = {
        'students_with_paths': students_with_paths,
        'students_without_paths': students_without_paths,
        'total_students': Student.objects.count(),
        'total_paths': LearningPath.objects.count(),
    }
    
    return render(request, 'learning_paths/student_list.html', context)


def student_paths(request, student_id):
    """
    Level 2: Display all learning paths for a specific student.
    """
    student = get_object_or_404(Student, student_id=student_id)
    
    # Get all learning paths for this student
    learning_paths = LearningPath.objects.filter(student=student).order_by('-generated_at')
    
    # Calculate some stats
    total_paths = learning_paths.count()
    total_time = sum(path.total_estimated_time for path in learning_paths)
    avg_weak_topics = learning_paths.aggregate(avg=Avg('weak_topics_count'))['avg'] or 0
    
    context = {
        'student': student,
        'learning_paths': learning_paths,
        'total_paths': total_paths,
        'total_time': total_time,
        'avg_weak_topics': round(avg_weak_topics, 1),
    }
    
    return render(request, 'learning_paths/student_paths.html', context)


def path_detail(request, path_id):
    """
    Level 3: Display detailed learning path with amazing visualization.
    """
    learning_path = get_object_or_404(LearningPath, id=path_id)
    
    # Get related data
    weak_topics = WeakTopic.objects.filter(learning_path=learning_path).order_by('order')
    recommended_topics = RecommendedTopic.objects.filter(learning_path=learning_path).order_by('priority')
    
    # Prepare data for visualization
    path_data = {
        'nodes': [],
        'edges': [],
        'student_profile': learning_path.student_stats,
        'path_stats': {
            'total_time': learning_path.total_estimated_time,
            'weak_topics_count': learning_path.weak_topics_count,
            'recommended_topics_count': learning_path.recommended_topics_count,
            'created_at': learning_path.generated_at.isoformat(),
            'status': 'Active' if learning_path.generated_at > timezone.now() - timedelta(days=7) else 'Old'
        }
    }
    
    # Build nodes for visualization
    node_id = 0
    
    # Add start node
    path_data['nodes'].append({
        'id': node_id,
        'label': 'START',
        'type': 'start',
        'color': '#28a745',
        'size': 30,
        'font': {'size': 16, 'color': 'white'}
    })
    start_node_id = node_id
    node_id += 1
    
    # Add recommended topic nodes
    prev_node_id = start_node_id
    topic_nodes = {}
    
    for rec_topic in recommended_topics[:8]:  # Limit to 8 for better visualization
        # Determine node color based on prerequisites
        if rec_topic.should_study_prerequisites_first:
            color = '#ffc107'  # Yellow for prerequisites needed
            status = 'Prerequisites Needed'
        else:
            color = '#007bff'  # Blue for ready to study
            status = 'Ready to Study'
        
        # Determine size based on confidence
        size = 20 + (rec_topic.confidence * 20)  # Size between 20-40
        
        path_data['nodes'].append({
            'id': node_id,
            'label': rec_topic.topic.name[:20] + ('...' if len(rec_topic.topic.name) > 20 else ''),
            'full_name': rec_topic.topic.name,
            'type': 'topic',
            'color': color,
            'size': size,
            'confidence': round(rec_topic.confidence * 100, 1),
            'time_hours': rec_topic.estimated_time_hours,
            'difficulty': rec_topic.recommended_difficulty,
            'status': status,
            'prerequisites': rec_topic.prerequisites,
            'unmet_prerequisites': rec_topic.unmet_prerequisites,
            'resources': [
                {
                    'title': res.title,
                    'url': res.url,
                    'type': res.resource_type,
                    'difficulty': res.difficulty,
                    'time': res.estimated_time
                }
                for res in TopicResource.objects.filter(recommended_topic=rec_topic).order_by('order')[:5]
            ],
            'font': {'size': 12, 'color': 'white'}
        })
        
        # Add edge from previous node
        path_data['edges'].append({
            'from': prev_node_id,
            'to': node_id,
            'arrows': 'to',
            'color': {'color': '#666666'},
            'width': 2
        })
        
        topic_nodes[rec_topic.topic.name] = node_id
        prev_node_id = node_id
        node_id += 1
    
    # Add finish node
    path_data['nodes'].append({
        'id': node_id,
        'label': 'FINISH',
        'type': 'finish',
        'color': '#dc3545',
        'size': 30,
        'font': {'size': 16, 'color': 'white'}
    })
    
    # Add edge to finish
    if prev_node_id != start_node_id:
        path_data['edges'].append({
            'from': prev_node_id,
            'to': node_id,
            'arrows': 'to',
            'color': {'color': '#666666'},
            'width': 2
        })
    
    context = {
        'learning_path': learning_path,
        'weak_topics': weak_topics,
        'recommended_topics': recommended_topics,
        'path_data_json': json.dumps(path_data),
        'student_profile': learning_path.student_stats,
    }
    
    return render(request, 'learning_paths/path_detail.html', context)


def generate_new_path(request, student_id):
    """
    Generate a new learning path for a student.
    """
    if request.method == 'POST':
        try:
            # Generate new learning path
            path_generator = DjangoIntegratedPathGenerator()
            learning_path_data = path_generator.generate_comprehensive_learning_path(student_id)
            
            if learning_path_data:
                messages.success(request, f'New learning path generated for student {student_id}!')
                # Redirect to the student's paths page
                return redirect('learning_paths:student_paths', student_id=student_id)
            else:
                messages.error(request, 'Failed to generate learning path. Please try again.')
        except Exception as e:
            messages.error(request, f'Error generating learning path: {str(e)}')
    
    return redirect('learning_paths:student_paths', student_id=student_id)


# Keep existing views for backward compatibility
def learning_path_dashboard(request):
    """Original dashboard view."""
    return redirect('learning_paths:student_list')


def generate_learning_path(request, student_id):
    """Original generate view."""
    return generate_new_path(request, student_id)


def topic_resources(request, topic_id):
    """Get resources for a specific recommended topic."""
    try:
        recommended_topic = get_object_or_404(RecommendedTopic, id=topic_id)
        resources = TopicResource.objects.filter(recommended_topic=recommended_topic).order_by('order')
        
        resources_data = []
        for resource in resources:
            resources_data.append({
                'title': resource.title,
                'description': resource.description,
                'url': resource.url,
                'resource_type': resource.resource_type,
                'difficulty': resource.difficulty,
                'estimated_time': resource.estimated_time,
            })
        
        return JsonResponse({
            'topic': recommended_topic.topic.name,
            'resources': resources_data
        })
        
    except Exception as e:
        return JsonResponse({
            'error': f'Error loading resources: {str(e)}'
        }, status=500)


def test_integrated_models(request):
    """Test the integrated models."""
    try:
        path_generator = DjangoIntegratedPathGenerator()
        
        # Test with first student
        first_student = Student.objects.first()
        if first_student:
            test_path = path_generator.generate_comprehensive_learning_path(first_student.student_id)
            
            if test_path:
                return JsonResponse({
                    'status': 'success',
                    'message': 'Models are working correctly',
                    'test_student': first_student.student_id,
                    'path_topics': len(test_path.get('recommended_path', []))
                })
            else:
                return JsonResponse({
                    'status': 'error',
                    'message': 'Failed to generate test path'
                })
        else:
            return JsonResponse({
                'status': 'error',
                'message': 'No students found in database'
            })
            
    except Exception as e:
        return JsonResponse({
            'status': 'error',
            'message': f'Error testing models: {str(e)}'
        })
