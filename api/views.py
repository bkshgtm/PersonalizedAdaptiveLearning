from django.shortcuts import get_object_or_404
from django.views.decorators.cache import cache_page
import datetime
from django.utils import timezone

from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework import status

from core.models import Student, Course, Topic, Question, StudentInteraction
from ml_models.models import KnowledgeTracingModel, TopicMastery, PredictionBatch
from knowledge_graph.models import KnowledgeGraph
from learning_paths.models import LearningPath, PathGenerationJob, PathGenerator, LearningPathItem

from ml_models.tasks import generate_mastery_predictions
from learning_paths.tasks import generate_learning_path




@api_view(['GET'])
@permission_classes([IsAuthenticated])
def student_profile(request, student_id):
    """
    API endpoint to get a student's profile with knowledge states and learning path.
    
    Parameters:
    - student_id: The ID of the student
    - course_id: Query parameter for the course ID
    
    Returns:
    - Student information
    - Course information
    - Knowledge states for topics
    - Active learning path if available
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
    
    # Get course topics
    topics = Topic.objects.filter(course=course)
    
    # Get knowledge states
    latest_batch = PredictionBatch.objects.filter(
        model__course=course,
        status='completed'
    ).order_by('-completed_at').first()
    
    knowledge_states = []
    
    if latest_batch:
        masteries = TopicMastery.objects.filter(
            student=student,
            prediction_batch=latest_batch,
            topic__in=topics
        ).select_related('topic')
        
        for mastery in masteries:
            knowledge_states.append({
                "topic_id": mastery.topic.id,
                "topic_name": mastery.topic.name,
                "mastery_score": mastery.mastery_score,
                "trend": mastery.trend,
                "confidence": mastery.confidence
            })
    
    # Get active learning path
    learning_path = LearningPath.objects.filter(
        student=student,
        course=course,
        status='active'
    ).order_by('-generated_at').first()
    
    path_info = None
    if learning_path:
        path_items = []
        for item in learning_path.items.all().order_by('priority'):
            path_items.append({
                "id": item.id,
                "topic_id": item.topic.id,
                "topic_name": item.topic.name,
                "priority": item.priority,
                "status": item.status,
                "completed": item.completed
            })
            
        path_info = {
            "id": learning_path.id,
            "name": learning_path.name,
            "generated_at": learning_path.generated_at,
            "overall_progress": learning_path.overall_progress,
            "items": path_items
        }
    
    return Response({
        "student": {
            "id": student.student_id,
            "academic_level": student.academic_level,
            "major": student.major,
            "gpa": student.gpa,
            "study_frequency": student.study_frequency
        },
        "course": {
            "id": course.course_id,
            "title": course.title
        },
        "knowledge_states": knowledge_states,
        "learning_path": path_info
    })
    



@api_view(['POST'])
@permission_classes([IsAuthenticated])
def generate_recommendations(request, student_id):
    """
    API endpoint to generate knowledge states and learning path for a student.
    
    Parameters:
    - student_id: The ID of the student
    - course_id: The ID of the course (in request body)
    
    Returns:
    - Status of the generation process
    - IDs for the created jobs
    """
    course_id = request.data.get('course_id')
    
    if not course_id:
        return Response(
            {"error": "course_id is required."},
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
    
    # Step 1: Generate knowledge states
    
    # Get the default model for the course
    model = KnowledgeTracingModel.objects.filter(
        course=course, 
        is_default=True, 
        status='active'
    ).first()
    
    if not model:
        return Response(
            {"error": "No active knowledge tracing model found for this course."},
            status=status.HTTP_400_BAD_REQUEST
        )
    
    # Create a prediction batch
    prediction_batch = PredictionBatch.objects.create(
        model=model,
        status='pending'
    )
    
    # Start the prediction task
    generate_mastery_predictions.delay(prediction_batch.id)
    
    # Step 2: Generate learning path
    
    # Get the active path generator
    generator = PathGenerator.objects.filter(is_active=True).first()
    
    if not generator:
        return Response(
            {"error": "No active path generator found."},
            status=status.HTTP_400_BAD_REQUEST
        )
    
    # Get the active knowledge graph
    knowledge_graph = KnowledgeGraph.objects.filter(is_active=True).first()
    
    # Create a path generation job
    job = PathGenerationJob.objects.create(
        generator=generator,
        student=student,
        course=course,
        status='pending',
        prediction_batch=prediction_batch,
        knowledge_graph=knowledge_graph
    )
    
    # Start the generation task
    generate_learning_path.delay(job.id)
    
    return Response({
        "message": "Started generating recommendations for the student.",
        "prediction_batch_id": prediction_batch.id,
        "job_id": job.id
    }, status=status.HTTP_202_ACCEPTED)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def check_recommendation_status(request):
    """
    API endpoint to check the status of recommendation generation.
    
    Parameters:
    - prediction_batch_id: ID of the prediction batch (query parameter)
    - job_id: ID of the path generation job (query parameter)
    
    Returns:
    - Status of both the prediction batch and path generation job
    - Whether the entire process is completed
    - Path ID if available
    """
    prediction_batch_id = request.query_params.get('prediction_batch_id')
    job_id = request.query_params.get('job_id')
    
    if not prediction_batch_id or not job_id:
        return Response(
            {"error": "prediction_batch_id and job_id are required."},
            status=status.HTTP_400_BAD_REQUEST
        )
    
    try:
        prediction_batch = PredictionBatch.objects.get(pk=prediction_batch_id)
        job = PathGenerationJob.objects.get(pk=job_id)
    except (PredictionBatch.DoesNotExist, PathGenerationJob.DoesNotExist):
        return Response(
            {"error": "Prediction batch or job not found."},
            status=status.HTTP_404_NOT_FOUND
        )
    
    # Check if both are completed
    prediction_complete = prediction_batch.status == 'completed'
    job_complete = job.status == 'completed'
    
    # Check if there's a learning path
    path = None
    try:
        if job_complete:
            path = job.path
    except LearningPath.DoesNotExist:
        pass
    
    return Response({
        "prediction_status": prediction_batch.status,
        "job_status": job.status,
        "completed": prediction_complete and job_complete,
        "path_id": path.id if path else None
    })


@api_view(['GET'])
@permission_classes([IsAuthenticated])
@cache_page(60)  # Cache for 60 seconds
def monitoring_dashboard(request):
    """
    API endpoint to get system monitoring data.
    
    Returns:
    - Counts for various models
    - Recent activity
    """
    # Get counts for various models
    students_count = Student.objects.count()
    courses_count = Course.objects.count()
    topics_count = Topic.objects.count()
    
    # ML model stats
    ml_models_count = KnowledgeTracingModel.objects.count()
    active_ml_models = KnowledgeTracingModel.objects.filter(status='active').count()
    prediction_batches = PredictionBatch.objects.count()
    
    # Learning path stats
    learning_paths_count = LearningPath.objects.count()
    active_paths = LearningPath.objects.filter(status='active').count()
    completed_paths = LearningPath.objects.filter(status='completed').count()
    
    # Recent activity
    recent_predictions = PredictionBatch.objects.filter(
        status='completed'
    ).order_by('-completed_at')[:5].values('id', 'model__name', 'completed_at')
    
    recent_paths = LearningPath.objects.order_by(
        '-generated_at'
    )[:5].values('id', 'student__student_id', 'course__title', 'generated_at')
    
    return Response({
        "counts": {
            "students": students_count,
            "courses": courses_count,
            "topics": topics_count,
            "ml_models": ml_models_count,
            "active_ml_models": active_ml_models,
            "prediction_batches": prediction_batches,
            "learning_paths": learning_paths_count,
            "active_paths": active_paths,
            "completed_paths": completed_paths
        },
        "recent_activity": {
            "predictions": list(recent_predictions),
            "paths": list(recent_paths)
        }
    })


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def recommendation_dashboard(request, course_id):
    """
    API endpoint to get course-wide recommendation statistics.
    
    Parameters:
    - course_id: The ID of the course
    
    Returns:
    - Topic-level statistics
    - Learning path statistics
    """
    try:
        course = Course.objects.get(course_id=course_id)
    except Course.DoesNotExist:
        return Response(
            {"error": "Course not found."},
            status=status.HTTP_404_NOT_FOUND
        )
    
    # Get topics for this course
    topics = Topic.objects.filter(course=course)
    
    # Get latest prediction batch
    latest_batch = PredictionBatch.objects.filter(
        model__course=course,
        status='completed'
    ).order_by('-completed_at').first()
    
    if not latest_batch:
        return Response(
            {"error": "No prediction data available for this course."},
            status=status.HTTP_404_NOT_FOUND
        )
    
    # Calculate topic statistics
    topic_stats = []
    
    for topic in topics:
        # Get masteries for this topic
        masteries = TopicMastery.objects.filter(
            topic=topic,
            prediction_batch=latest_batch
        )
        
        if not masteries:
            continue
        
        # Calculate stats
        from django.db.models import Avg, Count
        mastery_stats = masteries.aggregate(
            avg_mastery=Avg('mastery_score'),
            count=Count('id')
        )
        
        # Skip topics with no data
        if mastery_stats['count'] == 0:
            continue
            
        # Get status counts
        weak_count = masteries.filter(mastery_score__lt=0.4).count()
        developing_count = masteries.filter(mastery_score__gte=0.4, mastery_score__lt=0.7).count()
        strong_count = masteries.filter(mastery_score__gte=0.7).count()
        
        # Get trend counts
        improving_count = masteries.filter(trend='improving').count()
        declining_count = masteries.filter(trend='declining').count()
        stagnant_count = masteries.filter(trend='stagnant').count()
        
        topic_stats.append({
            "topic_id": topic.id,
            "topic_name": topic.name,
            "average_mastery": mastery_stats['avg_mastery'],
            "student_counts": {
                "weak": weak_count,
                "developing": developing_count,
                "strong": strong_count
            },
            "trend_counts": {
                "improving": improving_count,
                "declining": declining_count,
                "stagnant": stagnant_count
            }
        })
    
    # Calculate learning path statistics
    active_paths = LearningPath.objects.filter(
        course=course,
        status='active'
    ).count()
    
    completed_paths = LearningPath.objects.filter(
        course=course,
        status='completed'
    ).count()
    
    # Most recommended topics (appear most often in learning paths)
    from django.db.models import Count
    popular_topics = LearningPathItem.objects.filter(
        path__course=course
    ).values('topic__name').annotate(
        count=Count('topic')
    ).order_by('-count')[:5]
    
    return Response({
        "course": {
            "id": course.course_id,
            "title": course.title
        },
        "topic_stats": topic_stats,
        "path_stats": {
            "active_paths": active_paths,
            "completed_paths": completed_paths,
            "popular_topics": list(popular_topics)
        },
        "last_updated": latest_batch.completed_at
    })


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def api_documentation(request):
    """
    API endpoint to provide documentation for the API.
    
    Returns:
    - List of available endpoints with descriptions
    """
    endpoints = [
        {
            "path": "/api/auth/token/",
            "method": "POST",
            "description": "Obtain an authentication token",
            "parameters": ["username", "password"]
        },
        {
            "path": "/api/students/{student_id}/profile/",
            "method": "GET",
            "description": "Get a student's profile with knowledge states and learning path",
            "parameters": ["student_id", "course_id (query)"]
        },
        {
            "path": "/api/students/{student_id}/recommendations/",
            "method": "POST",
            "description": "Generate knowledge states and learning path for a student",
            "parameters": ["student_id", "course_id (body)"]
        },
        {
            "path": "/api/recommendation-status/",
            "method": "GET",
            "description": "Check the status of recommendation generation",
            "parameters": ["prediction_batch_id (query)", "job_id (query)"]
        },
        {
            "path": "/api/monitoring/dashboard/",
            "method": "GET",
            "description": "Get system monitoring data",
            "parameters": []
        },
        {
            "path": "/api/courses/{course_id}/recommendations/",
            "method": "GET",
            "description": "Get course-wide recommendation statistics",
            "parameters": ["course_id"]
        },
        {
            "path": "/api/data-ingestion/upload/",
            "method": "POST",
            "description": "Upload a CSV file with student interaction data",
            "parameters": ["file"]
        },
        {
            "path": "/api/knowledge-graph/active/",
            "method": "GET",
            "description": "Get the active knowledge graph",
            "parameters": []
        },
        {
            "path": "/api/ml-models/models/",
            "method": "GET",
            "description": "Get all knowledge tracing models for a course",
            "parameters": ["course_id (query)"]
        },
        {
            "path": "/api/learning-paths/student/{student_id}/path/",
            "method": "GET",
            "description": "Get a student's active learning path",
            "parameters": ["student_id", "course_id (query)"]
        }
    ]
    
    return Response({
        "api_version": "1.0",
        "base_url": "/api/",
        "authentication": "Token-based authentication using Authorization header",
        "endpoints": endpoints
    })


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def validate_answer_api(request):
    """
    API endpoint to validate a student's answer using DeepSeek.
    """
    question_id = request.data.get('question_id')
    student_id = request.data.get('student_id')
    student_answer = request.data.get('answer')
    
    if not question_id or not student_id or not student_answer:
        return Response(
            {"error": "question_id, student_id, and answer are required."},
            status=status.HTTP_400_BAD_REQUEST
        )
    
    try:
        question = Question.objects.get(question_id=question_id)
        student = Student.objects.get(student_id=student_id)
    except (Question.DoesNotExist, Student.DoesNotExist):
        return Response(
            {"error": "Question or student not found."},
            status=status.HTTP_404_NOT_FOUND
        )
    
    # Validate the answer
    from core.services.answer_validation import AnswerValidator
    validator = AnswerValidator()
    result = validator.validate_answer(question, student_answer)
    
    # Record the interaction
    interaction = StudentInteraction.objects.create(
        student=student,
        question=question,
        response=student_answer,
        correct=result['is_correct'],
        score=result['score'],
        time_taken=datetime.timedelta(seconds=int(request.data.get('time_taken', 0))),
        timestamp=timezone.now(),
        attempt_number=int(request.data.get('attempt_number', 1)),
        resource_viewed_before=request.data.get('resource_viewed_before', False)
    )
    
    return Response({
        "validation": result,
        "interaction_id": interaction.id
    })
