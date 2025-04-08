from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.http import JsonResponse
from django.views.decorators.http import require_POST

from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework import status

from .models import ClassificationModel, ClassificationJob, ClassificationResult
from .tasks import classify_questions, classify_new_questions
from core.models import Question, Topic


@login_required
def model_list(request):
    """
    View to show all classification models.
    """
    models = ClassificationModel.objects.all().order_by('-updated_at')
    return render(request, 'topic_classification/model_list.html', {'models': models})


@login_required
def model_detail(request, model_id):
    """
    View to show details of a classification model.
    """
    model = get_object_or_404(ClassificationModel, pk=model_id)
    jobs = model.jobs.all().order_by('-created_at')[:10]
    
    return render(request, 'topic_classification/model_detail.html', {
        'model': model,
        'jobs': jobs
    })


@login_required
def create_model(request):
    """
    View to create a new classification model.
    """
    if request.method == 'POST':
        model_type = request.POST.get('model_type')
        name = request.POST.get('name')
        description = request.POST.get('description')
        is_default = request.POST.get('is_default') == 'on'
        
        if not name or not model_type:
            messages.error(request, "Model name and type are required.")
            return redirect('create_model')
        
        # If this model is set as default, unset other default models
        if is_default:
            ClassificationModel.objects.filter(is_default=True).update(is_default=False)
        
        # Create model path
        import os
        import uuid
        from django.conf import settings
        
        model_filename = f"{uuid.uuid4()}.pkl"
        model_path = os.path.join(settings.MODEL_STORAGE_PATH, 'topic_classification', model_filename)
        
        # Create the model
        model = ClassificationModel.objects.create(
            name=name,
            model_type=model_type,
            description=description,
            created_by=request.user,
            model_path=model_path,
            is_default=is_default,
            status='inactive'
        )
        
        messages.success(request, f"Classification model '{name}' created successfully.")
        return redirect('model_detail', model_id=model.id)
    
    return render(request, 'topic_classification/create_model.html')


@require_POST
@login_required
def set_default_model(request, model_id):
    """
    View to set a model as the default.
    """
    model = get_object_or_404(ClassificationModel, pk=model_id)
    
    # Unset other default models
    ClassificationModel.objects.filter(is_default=True).update(is_default=False)
    
    # Set this model as default
    model.is_default = True
    model.save(update_fields=['is_default'])
    
    messages.success(request, f"Model '{model.name}' is now the default classifier.")
    return redirect('model_list')


@require_POST
@login_required
def train_model(request, model_id):
    """
    View to train/initialize a classification model.
    """
    model = get_object_or_404(ClassificationModel, pk=model_id)
    
    # Create directory if it doesn't exist
    import os
    os.makedirs(os.path.dirname(model.model_path), exist_ok=True)
    
    # Update model status
    model.status = 'training'
    model.save(update_fields=['status'])
    
    try:
        # Initialize the classifier (which trains the model)
        from .services.classifier import _get_classifier
        classifier = _get_classifier(model.id)
        
        messages.success(request, f"Model '{model.name}' is being trained. This may take a few minutes.")
    except Exception as e:
        model.status = 'failed'
        model.save(update_fields=['status'])
        messages.error(request, f"Error training model: {str(e)}")
    
    return redirect('model_detail', model_id=model.id)


@login_required
def job_list(request):
    """
    View to show all classification jobs.
    """
    jobs = ClassificationJob.objects.all().order_by('-created_at')
    return render(request, 'topic_classification/job_list.html', {'jobs': jobs})


@login_required
def job_detail(request, job_id):
    """
    View to show details of a classification job.
    """
    job = get_object_or_404(ClassificationJob, pk=job_id)
    results = job.results.all().order_by('-classified_at')[:100]
    
    return render(request, 'topic_classification/job_detail.html', {
        'job': job,
        'results': results
    })


@require_POST
@login_required
def start_classification(request, model_id):
    """
    View to start a new classification job.
    """
    model = get_object_or_404(ClassificationModel, pk=model_id)
    
    # Check if the model is active
    if model.status != 'active':
        messages.error(request, f"Model '{model.name}' is not active. Please train it first.")
        return redirect('model_detail', model_id=model.id)
    
    # Find questions without topics
    questions = Question.objects.filter(topic__isnull=True)
    question_count = questions.count()
    
    if question_count == 0:
        messages.info(request, "No new questions to classify.")
        return redirect('model_detail', model_id=model.id)
    
    # Create a new classification job
    job = ClassificationJob.objects.create(
        model=model,
        status='pending',
        total_questions=question_count
    )
    
    # Start the classification task
    classify_questions.delay(job.id)
    
    messages.success(
        request,
        f"Started classification job for {question_count} questions. "
        f"You can check the status on the job detail page."
    )
    
    return redirect('job_detail', job_id=job.id)


@require_POST
@login_required
def verify_classification(request, result_id):
    """
    View to manually verify a classification result.
    """
    result = get_object_or_404(ClassificationResult, pk=result_id)
    
    # Mark as verified
    result.is_verified = True
    result.verified_by = request.user
    result.save(update_fields=['is_verified', 'verified_by'])
    
    return JsonResponse({'status': 'success'})


@require_POST
@login_required
def update_classification(request, result_id):
    """
    View to update a classification result with a different topic.
    """
    result = get_object_or_404(ClassificationResult, pk=result_id)
    
    # Get the new topic
    topic_id = request.POST.get('topic_id')
    if not topic_id:
        return JsonResponse({
            'status': 'error',
            'message': 'Topic ID is required'
        })
    
    try:
        topic = Topic.objects.get(pk=topic_id)
    except Topic.DoesNotExist:
        return JsonResponse({
            'status': 'error',
            'message': 'Topic not found'
        })
    
    # Update the result
    result.topic = topic
    result.is_verified = True
    result.verified_by = request.user
    result.save(update_fields=['topic', 'is_verified', 'verified_by'])
    
    # Update the question's topic
    question = result.question
    question.topic = topic
    question.save(update_fields=['topic'])
    
    return JsonResponse({'status': 'success'})


# API Views
@api_view(['POST'])
@permission_classes([IsAuthenticated])
def create_model_api(request):
    """
    API endpoint to create a new classification model.
    """
    name = request.data.get('name')
    model_type = request.data.get('model_type')
    description = request.data.get('description', '')
    is_default = request.data.get('is_default', False)
    
    if not name or not model_type:
        return Response(
            {"error": "Model name and type are required."},
            status=status.HTTP_400_BAD_REQUEST
        )
    
    # If this model is set as default, unset other default models
    if is_default:
        ClassificationModel.objects.filter(is_default=True).update(is_default=False)
    
    # Create model path
    import os
    import uuid
    from django.conf import settings
    
    model_filename = f"{uuid.uuid4()}.pkl"
    model_path = os.path.join(settings.MODEL_STORAGE_PATH, 'topic_classification', model_filename)
    
    # Create the model
    model = ClassificationModel.objects.create(
        name=name,
        model_type=model_type,
        description=description,
        created_by=request.user,
        model_path=model_path,
        is_default=is_default,
        status='inactive'
    )
    
    return Response({
        "id": model.id,
        "name": model.name,
        "model_type": model.model_type,
        "status": model.status,
        "is_default": model.is_default
    }, status=status.HTTP_201_CREATED)


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def train_model_api(request, model_id):
    """
    API endpoint to train a classification model.
    """
    try:
        model = ClassificationModel.objects.get(pk=model_id)
    except ClassificationModel.DoesNotExist:
        return Response(
            {"error": "Model not found."},
            status=status.HTTP_404_NOT_FOUND
        )
    
    # Create directory if it doesn't exist
    import os
    os.makedirs(os.path.dirname(model.model_path), exist_ok=True)
    
    # Update model status
    model.status = 'training'
    model.save(update_fields=['status'])
    
    try:
        # Initialize the classifier (which trains the model)
        from .services.classifier import _get_classifier
        classifier = _get_classifier(model.id)
        
        return Response({
            "message": f"Model '{model.name}' is being trained.",
            "id": model.id,
            "status": model.status
        })
    except Exception as e:
        model.status = 'failed'
        model.save(update_fields=['status'])
        
        return Response(
            {"error": f"Error training model: {str(e)}"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def start_classification_api(request, model_id):
    """
    API endpoint to start a classification job.
    """
    try:
        model = ClassificationModel.objects.get(pk=model_id)
    except ClassificationModel.DoesNotExist:
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
    
    # Find questions without topics
    questions = Question.objects.filter(topic__isnull=True)
    question_count = questions.count()
    
    if question_count == 0:
        return Response(
            {"message": "No new questions to classify."},
            status=status.HTTP_200_OK
        )
    
    # Create a new classification job
    job = ClassificationJob.objects.create(
        model=model,
        status='pending',
        total_questions=question_count
    )
    
    # Start the classification task
    classify_questions.delay(job.id)
    
    return Response({
        "message": f"Started classification job for {question_count} questions.",
        "job_id": job.id,
        "status": job.status,
        "total_questions": question_count
    }, status=status.HTTP_202_ACCEPTED)
