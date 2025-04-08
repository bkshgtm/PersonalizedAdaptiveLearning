from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.http import JsonResponse
from django.utils import timezone
from django.views.decorators.http import require_POST

from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework import status

from .models import (
    PathGenerator, PathGenerationJob, LearningPath, 
    LearningPathItem, LearningResource, PathCheckpoint
)
from .tasks import generate_learning_path, refresh_learning_paths
from core.models import Student, Course, Topic


@login_required
def generator_list(request):
    """
    View to show all path generators.
    """
    generators = PathGenerator.objects.all().order_by('-updated_at')
    return render(request, 'learning_paths/generator_list.html', {'generators': generators})


@login_required
def generator_detail(request, generator_id):
    """
    View to show details of a path generator.
    """
    generator = get_object_or_404(PathGenerator, pk=generator_id)
    jobs = generator.jobs.all().order_by('-created_at')[:10]
    
    return render(request, 'learning_paths/generator_detail.html', {
        'generator': generator,
        'jobs': jobs
    })


@login_required
def create_generator(request):
    """
    View to create a new path generator.
    """
    if request.method == 'POST':
        name = request.POST.get('name')
        description = request.POST.get('description', '')
        is_active = request.POST.get('is_active') == 'on'
        
        # Parse configuration from form
        include_strong_topics = request.POST.get('include_strong_topics') == 'on'
        max_resources = int(request.POST.get('max_resources_per_topic', 3))
        
        config = {
            'include_strong_topics': include_strong_topics,
            'max_resources_per_topic': max_resources
        }
        
        if not name:
            messages.error(request, "Generator name is required.")
            return redirect('create_generator')
        
        # If this generator is set as active, unset other active generators
        if is_active:
            PathGenerator.objects.filter(is_active=True).update(is_active=False)
        
        # Create the generator
        generator = PathGenerator.objects.create(
            name=name,
            description=description,
            created_by=request.user,
            is_active=is_active,
            config=config
        )
        
        messages.success(request, f"Path generator '{name}' created successfully.")
        return redirect('generator_detail', generator_id=generator.id)
    
    return render(request, 'learning_paths/create_generator.html')


@require_POST
@login_required
def set_active_generator(request, generator_id):
    """
    View to set a generator as active.
    """
    generator = get_object_or_404(PathGenerator, pk=generator_id)
    
    # Unset other active generators
    PathGenerator.objects.filter(is_active=True).update(is_active=False)
    
    # Set this generator as active
    generator.is_active = True
    generator.save(update_fields=['is_active'])
    
    messages.success(request, f"Generator '{generator.name}' is now the active generator.")
    return redirect('generator_list')


@login_required
def job_list(request):
    """
    View to show all path generation jobs.
    """
    jobs = PathGenerationJob.objects.all().order_by('-created_at')
    return render(request, 'learning_paths/job_list.html', {'jobs': jobs})


@login_required
def job_detail(request, job_id):
    """
    View to show details of a path generation job.
    """
    job = get_object_or_404(PathGenerationJob, pk=job_id)
    
    # Check if a path has been generated
    try:
        path = job.path
        return redirect('path_detail', path_id=path.id)
    except LearningPath.DoesNotExist:
        # No path generated yet
        pass
    
    return render(request, 'learning_paths/job_detail.html', {'job': job})


@require_POST
@login_required
def generate_path(request):
    """
    View to start a path generation job.
    """
    student_id = request.POST.get('student_id')
    course_id = request.POST.get('course_id')
    generator_id = request.POST.get('generator_id')
    
    if not student_id or not course_id:
        messages.error(request, "Student and course are required.")
        return redirect('job_list')
    
    try:
        student = Student.objects.get(student_id=student_id)
        course = Course.objects.get(course_id=course_id)
    except (Student.DoesNotExist, Course.DoesNotExist):
        messages.error(request, "Student or course not found.")
        return redirect('job_list')
    
    # Get generator
    if generator_id:
        try:
            generator = PathGenerator.objects.get(pk=generator_id)
        except PathGenerator.DoesNotExist:
            messages.error(request, "Selected generator not found.")
            return redirect('job_list')
    else:
        # Use active generator
        generator = PathGenerator.objects.filter(is_active=True).first()
        
        if not generator:
            messages.error(request, "No active path generator found.")
            return redirect('job_list')
    
    # Create the job
    job = PathGenerationJob.objects.create(
        generator=generator,
        student=student,
        course=course,
        status='pending'
    )
    
    # Start the generation task
    generate_learning_path.delay(job.id)
    
    messages.success(
        request,
        f"Started path generation job for student {student.student_id}. "
        f"The learning path will be ready shortly."
    )
    
    return redirect('job_detail', job_id=job.id)


@require_POST
@login_required
def refresh_course_paths(request, course_id):
    """
    View to refresh learning paths for all students in a course.
    """
    try:
        course = Course.objects.get(course_id=course_id)
    except Course.DoesNotExist:
        messages.error(request, "Course not found.")
        return redirect('job_list')
    
    # Start the refresh task
    refresh_learning_paths.delay(course_id)
    
    messages.success(
        request,
        f"Started learning path refresh for course {course.title}. "
        f"This may take some time for courses with many students."
    )
    
    return redirect('job_list')


@login_required
def path_list(request):
    """
    View to show all learning paths.
    """
    paths = LearningPath.objects.all().order_by('-generated_at')
    return render(request, 'learning_paths/path_list.html', {'paths': paths})


@login_required
def path_detail(request, path_id):
    """
    View to show details of a learning path.
    """
    path = get_object_or_404(LearningPath, pk=path_id)
    items = path.items.all().order_by('priority')
    checkpoints = path.checkpoints.all().order_by('position')
    
    return render(request, 'learning_paths/path_detail.html', {
        'path': path,
        'items': items,
        'checkpoints': checkpoints
    })


@login_required
def student_paths(request, student_id):
    """
    View to show all learning paths for a student.
    """
    student = get_object_or_404(Student, student_id=student_id)
    paths = LearningPath.objects.filter(student=student).order_by('-generated_at')
    
    return render(request, 'learning_paths/student_paths.html', {
        'student': student,
        'paths': paths
    })


@require_POST
@login_required
def mark_resource_viewed(request, resource_id):
    """
    View to mark a learning resource as viewed.
    """
    resource = get_object_or_404(LearningResource, pk=resource_id)
    
    # Mark as viewed
    resource.viewed = True
    resource.viewed_at = timezone.now()
    resource.save(update_fields=['viewed', 'viewed_at'])
    
    return JsonResponse({'status': 'success'})


@require_POST
@login_required
def mark_item_completed(request, item_id):
    """
    View to mark a learning path item as completed.
    """
    item = get_object_or_404(LearningPathItem, pk=item_id)
    
    # Mark as completed
    item.completed = True
    item.completed_at = timezone.now()
    item.save(update_fields=['completed', 'completed_at'])
    
    # Update path progress
    path = item.path
    completed_count = path.items.filter(completed=True).count()
    total_count = path.items.count()
    
    path.overall_progress.update({
        'completed_topics': completed_count,
        'in_progress_topics': 0,
        'not_started_topics': total_count - completed_count
    })
    path.save(update_fields=['overall_progress'])
    
    return JsonResponse({'status': 'success'})


@require_POST
@login_required
def mark_checkpoint_completed(request, checkpoint_id):
    """
    View to mark a path checkpoint as completed.
    """
    checkpoint = get_object_or_404(PathCheckpoint, pk=checkpoint_id)
    score = float(request.POST.get('score', 0))
    
    # Mark as completed
    checkpoint.completed = True
    checkpoint.completed_at = timezone.now()
    checkpoint.score = score
    checkpoint.save(update_fields=['completed', 'completed_at', 'score'])
    
    return JsonResponse({'status': 'success'})


@require_POST
@login_required
def archive_path(request, path_id):
    """
    View to archive a learning path.
    """
    path = get_object_or_404(LearningPath, pk=path_id)
    
    # Archive the path
    path.status = 'archived'
    path.save(update_fields=['status'])
    
    messages.success(request, f"Learning path '{path.name}' has been archived.")
    return redirect('path_list')


# API Views
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_student_path_api(request, student_id):
    """
    API endpoint to get a student's active learning path.
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
    
    # Get the latest active path
    path = LearningPath.objects.filter(
        student=student,
        course=course,
        status='active'
    ).order_by('-generated_at').first()
    
    if not path:
        return Response(
            {"error": "No active learning path found for this student and course."},
            status=status.HTTP_404_NOT_FOUND
        )
    
    # Get path items
    items = []
    for item in path.items.all().order_by('priority'):
        resources = []
        for resource in item.resources.all():
            resources.append({
                "id": resource.id,
                "title": resource.resource.title,
                "url": resource.resource.url,
                "type": resource.resource.resource_type,
                "difficulty": resource.resource.difficulty,
                "estimated_time": str(resource.resource.estimated_time),
                "match_reason": resource.match_reason,
                "viewed": resource.viewed
            })
        
        items.append({
            "id": item.id,
            "topic_id": item.topic.id,
            "topic_name": item.topic.name,
            "priority": item.priority,
            "status": item.status,
            "proficiency_score": item.proficiency_score,
            "trend": item.trend,
            "confidence_of_improvement": item.confidence_of_improvement,
            "reason": item.reason,
            "estimated_review_time": str(item.estimated_review_time),
            "completed": item.completed,
            "resources": resources
        })
    
    # Get checkpoints
    checkpoints = []
    for checkpoint in path.checkpoints.all().order_by('position'):
        topic_ids = [topic.id for topic in checkpoint.topics.all()]
        
        checkpoints.append({
            "id": checkpoint.id,
            "name": checkpoint.name,
            "description": checkpoint.description,
            "checkpoint_type": checkpoint.checkpoint_type,
            "position": checkpoint.position,
            "topic_ids": topic_ids,
            "completed": checkpoint.completed,
            "score": checkpoint.score
        })
    
    return Response({
        "id": path.id,
        "name": path.name,
        "description": path.description,
        "generated_at": path.generated_at,
        "expires_at": path.expires_at,
        "status": path.status,
        "overall_progress": path.overall_progress,
        "estimated_completion_time": str(path.estimated_completion_time),
        "items": items,
        "checkpoints": checkpoints
    })


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def generate_path_api(request):
    """
    API endpoint to generate a learning path for a student.
    """
    student_id = request.data.get('student_id')
    course_id = request.data.get('course_id')
    
    if not student_id or not course_id:
        return Response(
            {"error": "student_id and course_id are required."},
            status=status.HTTP_400_BAD_REQUEST
        )
    
    try:
        student = Student.objects.get(student_id=student_id)
        course = Course.objects.get(course_id=course_id)
    except (Student.DoesNotExist, Course.DoesNotExist):
        return Response(
            {"error": "Student or course not found."},
            status=status.HTTP_404_NOT_FOUND
        )
    
    # Get active generator
    generator = PathGenerator.objects.filter(is_active=True).first()
    
    if not generator:
        return Response(
            {"error": "No active path generator found."},
            status=status.HTTP_400_BAD_REQUEST
        )
    
    # Create the job
    job = PathGenerationJob.objects.create(
        generator=generator,
        student=student,
        course=course,
        status='pending'
    )
    
    # Start the generation task
    generate_learning_path.delay(job.id)
    
    return Response({
        "message": f"Started path generation job for student {student_id}.",
        "job_id": job.id
    }, status=status.HTTP_202_ACCEPTED)


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def update_resource_status_api(request, resource_id):
    """
    API endpoint to update a learning resource status.
    """
    try:
        resource = LearningResource.objects.get(pk=resource_id)
    except LearningResource.DoesNotExist:
        return Response(
            {"error": "Resource not found."},
            status=status.HTTP_404_NOT_FOUND
        )
    
    # Mark as viewed
    resource.viewed = True
    resource.viewed_at = timezone.now()
    resource.save(update_fields=['viewed', 'viewed_at'])
    
    return Response({"message": "Resource marked as viewed."})


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def update_item_status_api(request, item_id):
    """
    API endpoint to update a learning path item status.
    """
    try:
        item = LearningPathItem.objects.get(pk=item_id)
    except LearningPathItem.DoesNotExist:
        return Response(
            {"error": "Learning path item not found."},
            status=status.HTTP_404_NOT_FOUND
        )
    
    # Mark as completed
    item.completed = True
    item.completed_at = timezone.now()
    item.save(update_fields=['completed', 'completed_at'])
    
    # Update path progress
    path = item.path
    completed_count = path.items.filter(completed=True).count()
    total_count = path.items.count()
    
    path.overall_progress.update({
        'completed_topics': completed_count,
        'in_progress_topics': 0,
        'not_started_topics': total_count - completed_count
    })
    path.save(update_fields=['overall_progress'])
    
    return Response({"message": "Item marked as completed."})


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def update_checkpoint_status_api(request, checkpoint_id):
    """
    API endpoint to update a checkpoint status.
    """
    try:
        checkpoint = PathCheckpoint.objects.get(pk=checkpoint_id)
    except PathCheckpoint.DoesNotExist:
        return Response(
            {"error": "Checkpoint not found."},
            status=status.HTTP_404_NOT_FOUND
        )
    
    score = float(request.data.get('score', 0))
    
    # Mark as completed
    checkpoint.completed = True
    checkpoint.completed_at = timezone.now()
    checkpoint.score = score
    checkpoint.save(update_fields=['completed', 'completed_at', 'score'])
    
    return Response({"message": "Checkpoint marked as completed."})