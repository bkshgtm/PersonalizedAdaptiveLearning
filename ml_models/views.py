from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.http import JsonResponse
from django.views.decorators.http import require_POST
from django.core.paginator import Paginator

from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework import status

from .models import KnowledgeTracingModel, TrainingJob, PredictionBatch, TopicMastery
from .tasks import train_knowledge_tracing_model, generate_mastery_predictions
from core.models import Course, Student, Topic


@login_required
def model_list(request):
    """
    View to show all knowledge tracing models.
    """
    models = KnowledgeTracingModel.objects.all().order_by('-updated_at')
    
    # Group by course
    courses = Course.objects.all()
    course_models = {}
    
    for course in courses:
        course_models[course] = models.filter(course=course)
    
    return render(request, 'ml_models/model_list.html', {
        'course_models': course_models,
        'models': models
    })


@login_required
def model_detail(request, model_id):
    """
    View to show details of a knowledge tracing model.
    """
    model = get_object_or_404(KnowledgeTracingModel, pk=model_id)
    jobs = model.training_jobs.all().order_by('-created_at')[:5]
    prediction_batches = model.prediction_batches.all().order_by('-created_at')[:5]
    
    # Calculate training stats
    total_jobs = model.training_jobs.count()
    successful_jobs = model.training_jobs.filter(status='completed').count()
    
    # Calculate prediction stats
    total_batches = model.prediction_batches.count()
    successful_batches = model.prediction_batches.filter(status='completed').count()
    
    # Get latest metrics
    latest_metrics = {}
    latest_job = model.training_jobs.filter(status='completed').order_by('-completed_at').first()
    if latest_job:
        latest_metrics = latest_job.metrics
    
    return render(request, 'ml_models/model_detail.html', {
        'model': model,
        'jobs': jobs,
        'prediction_batches': prediction_batches,
        'total_jobs': total_jobs,
        'successful_jobs': successful_jobs,
        'total_batches': total_batches,
        'successful_batches': successful_batches,
        'latest_metrics': latest_metrics
    })


@login_required
def create_model(request):
    """
    View to create a new knowledge tracing model.
    """
    if request.method == 'POST':
        name = request.POST.get('name')
        model_type = request.POST.get('model_type')
        description = request.POST.get('description', '')
        course_id = request.POST.get('course')
        is_default = request.POST.get('is_default') == 'on'
        
        if not name or not model_type or not course_id:
            messages.error(request, "Name, model type, and course are required.")
            return redirect('create_model')
        
        try:
            course = Course.objects.get(pk=course_id)
        except Course.DoesNotExist:
            messages.error(request, "Selected course does not exist.")
            return redirect('create_model')
        
        # If this model is set as default, unset other default models for this course
        if is_default:
            KnowledgeTracingModel.objects.filter(course=course, is_default=True).update(is_default=False)
        
        # Create model path
        import os
        import uuid
        from django.conf import settings
        
        model_filename = f"{uuid.uuid4()}.pt"
        model_path = os.path.join(settings.MODEL_STORAGE_PATH, 'knowledge_tracing', course.course_id, model_filename)
        
        # Create the model
        model = KnowledgeTracingModel.objects.create(
            name=name,
            model_type=model_type,
            description=description,
            created_by=request.user,
            model_path=model_path,
            is_default=is_default,
            status='created',
            course=course
        )
        
        # Set hyperparameters based on model type
        if model_type == 'dkt':
            hyperparameters = {
                'hidden_size': int(request.POST.get('hidden_size', 100)),
                'num_layers': int(request.POST.get('num_layers', 1)),
                'dropout': float(request.POST.get('dropout', 0.2)),
                'batch_size': int(request.POST.get('batch_size', 32)),
                'learning_rate': float(request.POST.get('learning_rate', 0.001)),
                'num_epochs': int(request.POST.get('num_epochs', 10))
            }
        elif model_type == 'sakt':
            hyperparameters = {
                'd_model': int(request.POST.get('d_model', 64)),
                'n_heads': int(request.POST.get('n_heads', 8)),
                'dropout': float(request.POST.get('dropout', 0.2)),
                'batch_size': int(request.POST.get('batch_size', 32)),
                'learning_rate': float(request.POST.get('learning_rate', 0.001)),
                'num_epochs': int(request.POST.get('num_epochs', 10))
            }
        else:
            hyperparameters = {}
        
        # Set as model hyperparameters
        model.hyperparameters = hyperparameters
        model.save(update_fields=['hyperparameters'])
        
        messages.success(request, f"Knowledge tracing model '{name}' created successfully.")
        return redirect('model_detail', model_id=model.id)
    
    # GET request
    courses = Course.objects.all()
    
    return render(request, 'ml_models/create_model.html', {
        'courses': courses,
        'model_types': [
            ('dkt', 'Deep Knowledge Tracing'),
            ('sakt', 'Self-Attentive Knowledge Tracing')
        ]
    })


@require_POST
@login_required
def set_default_model(request, model_id):
    """
    View to set a model as the default.
    """
    model = get_object_or_404(KnowledgeTracingModel, pk=model_id)
    
    # Unset other default models for this course
    KnowledgeTracingModel.objects.filter(course=model.course, is_default=True).update(is_default=False)
    
    # Set this model as default
    model.is_default = True
    model.save(update_fields=['is_default'])
    
    messages.success(request, f"Model '{model.name}' is now the default for course '{model.course.title}'.")
    return redirect('model_list')


@require_POST
@login_required
def train_model(request, model_id):
    """
    View to start a training job for a model.
    """
    model = get_object_or_404(KnowledgeTracingModel, pk=model_id)
    
    # Get training parameters
    split_ratio = float(request.POST.get('split_ratio', 0.8))
    
    # Create the training job
    job = TrainingJob.objects.create(
        model=model,
        status='pending',
        hyperparameters=model.hyperparameters,
        split_ratio=split_ratio
    )
    
    # Update model status
    model.status = 'training'
    model.save(update_fields=['status'])
    
    # Start the training task
    train_knowledge_tracing_model.delay(job.id)
    
    messages.success(
        request,
        f"Started training job for model '{model.name}'. "
        f"You can check the status on the job detail page."
    )
    
    return redirect('job_detail', job_id=job.id)


@login_required
def job_detail(request, job_id):
    """
    View to show details of a training job.
    """
    job = get_object_or_404(TrainingJob, pk=job_id)
    
    # Get model information
    model = job.model
    
    # Format job metrics for display
    formatted_metrics = {}
    for key, value in job.metrics.items():
        if isinstance(value, float):
            formatted_metrics[key] = f"{value:.4f}"
        else:
            formatted_metrics[key] = value
    
    # Calculate duration if job has started and completed
    duration = None
    if job.started_at and job.completed_at:
        duration = job.completed_at - job.started_at
    
    return render(request, 'ml_models/job_detail.html', {
        'job': job,
        'model': model,
        'formatted_metrics': formatted_metrics,
        'duration': duration
    })


@login_required
def job_list(request):
    """
    View to show all training jobs.
    """
    jobs = TrainingJob.objects.all().order_by('-created_at')
    
    # Pagination
    paginator = Paginator(jobs, 20)
    page_number = request.GET.get('page')
    jobs_page = paginator.get_page(page_number)
    
    return render(request, 'ml_models/job_list.html', {'jobs': jobs_page})


@require_POST
@login_required
def generate_predictions(request, model_id):
    """
    View to start a prediction batch for a model.
    """
    model = get_object_or_404(KnowledgeTracingModel, pk=model_id)
    
    # Check if the model is active
    if model.status != 'active':
        messages.error(request, f"Model '{model.name}' is not active. Please train it first.")
        return redirect('model_detail', model_id=model.id)
    
    # Create the prediction batch
    batch = PredictionBatch.objects.create(
        model=model,
        status='pending'
    )
    
    # Start the prediction task
    generate_mastery_predictions.delay(batch.id)
    
    messages.success(
        request,
        f"Started prediction batch for model '{model.name}'. "
        f"You can check the status on the batch detail page."
    )
    
    return redirect('batch_detail', batch_id=batch.id)


@login_required
def batch_detail(request, batch_id):
    """
    View to show details of a prediction batch.
    """
    batch = get_object_or_404(PredictionBatch, pk=batch_id)
    
    # Get a sample of masteries
    masteries = batch.masteries.all().select_related('student', 'topic')[:100]
    
    # Calculate duration if batch has started and completed
    duration = None
    if batch.started_at and batch.completed_at:
        duration = batch.completed_at - batch.started_at
    
    # Calculate statistics
    total_masteries = batch.masteries.count()
    avg_mastery = None
    if total_masteries > 0:
        from django.db.models import Avg
        avg_mastery = batch.masteries.aggregate(avg=Avg('mastery_score'))['avg']
    
    # Get trend stats
    from django.db.models import Count
    trend_stats = batch.masteries.values('trend').annotate(count=Count('id'))
    
    return render(request, 'ml_models/batch_detail.html', {
        'batch': batch,
        'masteries': masteries,
        'duration': duration,
        'total_masteries': total_masteries,
        'avg_mastery': avg_mastery,
        'trend_stats': trend_stats
    })


@login_required
def batch_list(request):
    """
    View to show all prediction batches.
    """
    batches = PredictionBatch.objects.all().order_by('-created_at')
    
    # Pagination
    paginator = Paginator(batches, 20)
    page_number = request.GET.get('page')
    batches_page = paginator.get_page(page_number)
    
    return render(request, 'ml_models/batch_list.html', {'batches': batches_page})


@login_required
def student_masteries(request, student_id):
    """
    View to show a student's topic masteries.
    """
    student = get_object_or_404(Student, student_id=student_id)
    
    # Get the latest prediction batch for each model
    latest_batches = PredictionBatch.objects.filter(
        status='completed'
    ).order_by('model', '-completed_at').distinct('model')
    
    # Get masteries for each batch
    all_masteries = {}
    
    for batch in latest_batches:
        masteries = TopicMastery.objects.filter(
            student=student,
            prediction_batch=batch
        ).select_related('topic')
        
        all_masteries[batch.id] = {
            'batch': batch,
            'masteries': masteries
        }
    
    # Get courses for this student
    courses = student.courses.all()
    
    return render(request, 'ml_models/student_masteries.html', {
        'student': student,
        'all_masteries': all_masteries,
        'courses': courses
    })


@login_required
def topic_masteries(request, topic_id):
    """
    View to show a topic's masteries across students.
    """
    topic = get_object_or_404(Topic, pk=topic_id)
    
    # Get the latest prediction batch
    latest_batch = PredictionBatch.objects.filter(
        status='completed',
        model__course=topic.course
    ).order_by('-completed_at').first()
    
    if not latest_batch:
        messages.info(request, "No prediction batches available for this topic's course.")
        return render(request, 'ml_models/topic_masteries.html', {
            'topic': topic,
            'masteries': []
        })
    
    # Get masteries for this topic in the latest batch
    masteries = TopicMastery.objects.filter(
        topic=topic,
        prediction_batch=latest_batch
    ).select_related('student')
    
    # Calculate statistics
    from django.db.models import Avg, Count, StdDev
    stats = masteries.aggregate(
        avg=Avg('mastery_score'),
        count=Count('id'),
        std=StdDev('mastery_score')
    )
    
    # Get trend stats
    trend_stats = masteries.values('trend').annotate(count=Count('id'))
    
    return render(request, 'ml_models/topic_masteries.html', {
        'topic': topic,
        'batch': latest_batch,
        'masteries': masteries,
        'stats': stats,
        'trend_stats': trend_stats
    })


@login_required
def course_masteries(request, course_id):
    """
    View to show course-level mastery statistics.
    """
    course = get_object_or_404(Course, pk=course_id)
    
    # Get the latest prediction batch for this course
    latest_batch = PredictionBatch.objects.filter(
        status='completed',
        model__course=course
    ).order_by('-completed_at').first()
    
    if not latest_batch:
        messages.info(request, "No prediction batches available for this course.")
        return render(request, 'ml_models/course_masteries.html', {
            'course': course,
            'topics': [],
            'batch': None
        })
    
    # Get all topics for this course
    topics = Topic.objects.filter(course=course)
    
    # Get masteries for each topic
    topic_stats = []
    
    for topic in topics:
        # Get masteries for this topic
        topic_masteries = TopicMastery.objects.filter(
            topic=topic,
            prediction_batch=latest_batch
        )
        
        # Calculate statistics
        from django.db.models import Avg, Count
        stats = topic_masteries.aggregate(
            avg=Avg('mastery_score'),
            count=Count('id')
        )
        
        # Skip topics with no masteries
        if stats['count'] == 0:
            continue
        
        # Get trend stats
        trend_counts = topic_masteries.values('trend').annotate(count=Count('id'))
        
        # Format trend stats
        trends = {}
        for trend in trend_counts:
            trends[trend['trend']] = trend['count']
        
        topic_stats.append({
            'topic': topic,
            'avg_mastery': stats['avg'],
            'student_count': stats['count'],
            'trends': trends
        })
    
    return render(request, 'ml_models/course_masteries.html', {
        'course': course,
        'topic_stats': topic_stats,
        'batch': latest_batch
    })


# API Views
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_models_api(request):
    """
    API endpoint to get all knowledge tracing models for a course.
    """
    course_id = request.query_params.get('course_id')
    
    if not course_id:
        return Response(
            {"error": "course_id is required."},
            status=status.HTTP_400_BAD_REQUEST
        )
    
    try:
        course = Course.objects.get(course_id=course_id)
    except Course.DoesNotExist:
        return Response(
            {"error": "Course not found."},
            status=status.HTTP_404_NOT_FOUND
        )
    
    models = KnowledgeTracingModel.objects.filter(course=course)
    
    result = []
    for model in models:
        result.append({
            "id": model.id,
            "name": model.name,
            "model_type": model.model_type,
            "status": model.status,
            "is_default": model.is_default,
            "created_at": model.created_at,
            "hyperparameters": model.hyperparameters
        })
    
    return Response(result)


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def create_model_api(request):
    """
    API endpoint to create a new knowledge tracing model.
    """
    name = request.data.get('name')
    model_type = request.data.get('model_type')
    description = request.data.get('description', '')
    course_id = request.data.get('course_id')
    is_default = request.data.get('is_default', False)
    hyperparameters = request.data.get('hyperparameters', {})
    
    if not name or not model_type or not course_id:
        return Response(
            {"error": "Name, model type, and course_id are required."},
            status=status.HTTP_400_BAD_REQUEST
        )
    
    try:
        course = Course.objects.get(course_id=course_id)
    except Course.DoesNotExist:
        return Response(
            {"error": "Course not found."},
            status=status.HTTP_404_NOT_FOUND
        )
    
    # If this model is set as default, unset other default models for this course
    if is_default:
        KnowledgeTracingModel.objects.filter(course=course, is_default=True).update(is_default=False)
    
    # Create model path
    import os
    import uuid
    from django.conf import settings
    
    model_filename = f"{uuid.uuid4()}.pt"
    model_path = os.path.join(settings.MODEL_STORAGE_PATH, 'knowledge_tracing', course.course_id, model_filename)
    
    # Create the model
    model = KnowledgeTracingModel.objects.create(
        name=name,
        model_type=model_type,
        description=description,
        created_by=request.user,
        model_path=model_path,
        is_default=is_default,
        status='created',
        course=course,
        hyperparameters=hyperparameters
    )
    
    return Response({
        "id": model.id,
        "name": model.name,
        "model_type": model.model_type,
        "status": model.status,
        "is_default": model.is_default,
        "created_at": model.created_at,
        "hyperparameters": model.hyperparameters
    }, status=status.HTTP_201_CREATED)


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def train_model_api(request, model_id):
    """
    API endpoint to train a knowledge tracing model.
    """
    try:
        model = KnowledgeTracingModel.objects.get(pk=model_id)
    except KnowledgeTracingModel.DoesNotExist:
        return Response(
            {"error": "Model not found."},
            status=status.HTTP_404_NOT_FOUND
        )
    
    # Get training parameters
    split_ratio = float(request.data.get('split_ratio', 0.8))
    hyperparameters = request.data.get('hyperparameters')
    
    # Use custom hyperparameters if provided, otherwise use model's
    if not hyperparameters:
        hyperparameters = model.hyperparameters
    
    # Create the training job
    job = TrainingJob.objects.create(
        model=model,
        status='pending',
        hyperparameters=hyperparameters,
        split_ratio=split_ratio
    )
    
    # Update model status
    model.status = 'training'
    model.save(update_fields=['status'])
    
    # Start the training task
    train_knowledge_tracing_model.delay(job.id)
    
    return Response({
        "message": f"Started training job for model '{model.name}'.",
        "job_id": job.id
    }, status=status.HTTP_202_ACCEPTED)


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def generate_predictions_api(request, model_id):
    """
    API endpoint to generate predictions using a knowledge tracing model.
    """
    try:
        model = KnowledgeTracingModel.objects.get(pk=model_id)
    except KnowledgeTracingModel.DoesNotExist:
        return Response(
            {"error": "Model not found."},
            status=status.HTTP_404_NOT_FOUND
        )
    
    # Check if the model is active
    if model.status != 'active':
        return Response(
            {"error": f"Model '{model.name}' is not active. Please train it first."},
            status=status.HTTP_400_BAD_REQUEST
        )
    
    # Create the prediction batch
    batch = PredictionBatch.objects.create(
        model=model,
        status='pending'
    )
    
    # Start the prediction task
    generate_mastery_predictions.delay(batch.id)
    
    return Response({
        "message": f"Started prediction batch for model '{model.name}'.",
        "batch_id": batch.id
    }, status=status.HTTP_202_ACCEPTED)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_job_status_api(request, job_id):
    """
    API endpoint to get the status of a training job.
    """
    try:
        job = TrainingJob.objects.get(pk=job_id)
    except TrainingJob.DoesNotExist:
        return Response(
            {"error": "Job not found."},
            status=status.HTTP_404_NOT_FOUND
        )
    
    return Response({
        "id": job.id,
        "model_id": job.model.id,
        "model_name": job.model.name,
        "status": job.status,
        "created_at": job.created_at,
        "started_at": job.started_at,
        "completed_at": job.completed_at,
        "metrics": job.metrics,
        "error_message": job.error_message
    })


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_batch_status_api(request, batch_id):
    """
    API endpoint to get the status of a prediction batch.
    """
    try:
        batch = PredictionBatch.objects.get(pk=batch_id)
    except PredictionBatch.DoesNotExist:
        return Response(
            {"error": "Prediction batch not found."},
            status=status.HTTP_404_NOT_FOUND
        )
    
    return Response({
        "id": batch.id,
        "model_id": batch.model.id,
        "model_name": batch.model.name,
        "status": batch.status,
        "created_at": batch.created_at,
        "started_at": batch.started_at,
        "completed_at": batch.completed_at,
        "total_students": batch.total_students,
        "processed_students": batch.processed_students,
        "error_message": batch.error_message
    })


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_student_masteries_api(request, student_id):
    """
    API endpoint to get a student's topic masteries.
    """
    try:
        student = Student.objects.get(student_id=student_id)
    except Student.DoesNotExist:
        return Response(
            {"error": "Student not found."},
            status=status.HTTP_404_NOT_FOUND
        )
    
    # Get course_id parameter
    course_id = request.query_params.get('course_id')
    
    if not course_id:
        return Response(
            {"error": "course_id parameter is required."},
            status=status.HTTP_400_BAD_REQUEST
        )
    
    try:
        course = Course.objects.get(course_id=course_id)
    except Course.DoesNotExist:
        return Response(
            {"error": "Course not found."},
            status=status.HTTP_404_NOT_FOUND
        )
    
    # Get the latest prediction batch for this course
    latest_batch = PredictionBatch.objects.filter(
        status='completed',
        model__course=course
    ).order_by('-completed_at').first()
    
    if not latest_batch:
        return Response(
            {"error": "No prediction data available for this course."},
            status=status.HTTP_404_NOT_FOUND
        )
    
    # Get masteries for this student and batch
    masteries = TopicMastery.objects.filter(
        student=student,
        prediction_batch=latest_batch
    ).select_related('topic')
    
    result = []
    for mastery in masteries:
        result.append({
            "topic_id": mastery.topic.id,
            "topic_name": mastery.topic.name,
            "mastery_score": mastery.mastery_score,
            "confidence": mastery.confidence,
            "trend": mastery.trend,
            "predicted_at": mastery.predicted_at
        })
    
    return Response({
        "student_id": student.student_id,
        "course_id": course.course_id,
        "batch_id": latest_batch.id,
        "predicted_at": latest_batch.completed_at,
        "masteries": result
    })


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_topic_masteries_api(request, topic_id):
    """
    API endpoint to get all masteries for a specific topic.
    """
    try:
        topic = Topic.objects.get(pk=topic_id)
    except Topic.DoesNotExist:
        return Response(
            {"error": "Topic not found."},
            status=status.HTTP_404_NOT_FOUND
        )
    
    # Get the latest prediction batch for this topic's course
    latest_batch = PredictionBatch.objects.filter(
        status='completed',
        model__course=topic.course
    ).order_by('-completed_at').first()
    
    if not latest_batch:
        return Response(
            {"error": "No prediction data available for this topic's course."},
            status=status.HTTP_404_NOT_FOUND
        )
    
    # Get masteries for this topic in the latest batch
    masteries = TopicMastery.objects.filter(
        topic=topic,
        prediction_batch=latest_batch
    ).select_related('student')
    
    # Calculate statistics
    from django.db.models import Avg, Count, StdDev
    stats = masteries.aggregate(
        avg_mastery=Avg('mastery_score'),
        student_count=Count('id'),
        std_deviation=StdDev('mastery_score')
    )
    
    # Get trend stats
    from django.db.models import Count
    trend_stats = list(masteries.values('trend').annotate(count=Count('id')))
    
    result = []
    for mastery in masteries:
        result.append({
            "student_id": mastery.student.student_id,
            "mastery_score": mastery.mastery_score,
            "confidence": mastery.confidence,
            "trend": mastery.trend
        })
    
    return Response({
        "topic_id": topic.id,
        "topic_name": topic.name,
        "course_id": topic.course.course_id,
        "batch_id": latest_batch.id,
        "predicted_at": latest_batch.completed_at,
        "statistics": stats,
        "trend_statistics": trend_stats,
        "masteries": result
    })