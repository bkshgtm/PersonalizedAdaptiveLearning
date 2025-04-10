from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.http import JsonResponse
from django.views.decorators.http import require_POST

from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework import status

from core.models import Student, Course, Assessment
from .models import DataUpload, ProcessingLog, DocumentUpload, ExtractedQuestion
from .services.data_validator import DataValidator
from .tasks import process_csv_upload, process_document_upload


@login_required
def upload_data(request):
    """
    View to handle data upload form.
    """
    if request.method == 'POST':
        if 'file' not in request.FILES:
            messages.error(request, "No file was uploaded.")
            return redirect('upload_data')
        
        uploaded_file = request.FILES['file']
        
        # Check file extension
        if not uploaded_file.name.endswith('.csv'):
            messages.error(request, "Only CSV files are accepted.")
            return redirect('upload_data')
        
        # Validate the CSV file
        validator = DataValidator(uploaded_file)
        is_valid, errors, warnings = validator.validate()
        
        if not is_valid:
            for error in errors:
                messages.error(request, error)
            return redirect('upload_data')
        
        # Create the data upload record
        data_upload = DataUpload.objects.create(
            file=uploaded_file,
            uploaded_by=request.user,
            status='pending'
        )
        
        # Log any warnings
        for warning in warnings:
            ProcessingLog.objects.create(
                data_upload=data_upload,
                message=warning,
                level='warning'
            )
        
        # Start the processing task
        process_csv_upload.delay(data_upload.id)
        
        messages.success(
            request,
            f"File '{uploaded_file.name}' has been uploaded and is being processed. "
            f"You can check the status at the upload detail page."
        )
        
        return redirect('upload_detail', upload_id=data_upload.id)
    
    return render(request, 'data_ingestion/upload_csv.html')


@login_required
def upload_list(request):
    """
    View to show a list of all data uploads.
    """
    uploads = DataUpload.objects.filter(uploaded_by=request.user).order_by('-uploaded_at')
    return render(request, 'data_ingestion/upload_list.html', {'uploads': uploads})


@login_required
def upload_detail(request, upload_id):
    """
    View to show details of a data upload, including logs.
    """
    upload = get_object_or_404(DataUpload, pk=upload_id, uploaded_by=request.user)
    logs = upload.logs.all().order_by('-timestamp')
    
    return render(request, 'data_ingestion/upload_detail.html', {
        'upload': upload,
        'logs': logs
    })


@require_POST
@login_required
def check_upload_status(request, upload_id):
    """
    AJAX endpoint to check the status of a data upload.
    """
    upload = get_object_or_404(DataUpload, pk=upload_id, uploaded_by=request.user)
    
    return JsonResponse({
        'status': upload.status,
        'rows_processed': upload.rows_processed,
        'rows_failed': upload.rows_failed,
        'error_message': upload.error_message
    })


# API Views
@api_view(['POST'])
@permission_classes([IsAuthenticated])
def upload_data_api(request):
    """
    API endpoint to upload data.
    """
    if 'file' not in request.FILES:
        return Response(
            {"error": "No file was uploaded."},
            status=status.HTTP_400_BAD_REQUEST
        )
    
    uploaded_file = request.FILES['file']
    
    # Check file extension
    if not uploaded_file.name.endswith('.csv'):
        return Response(
            {"error": "Only CSV files are accepted."},
            status=status.HTTP_400_BAD_REQUEST
        )
    
    # Validate the CSV file
    validator = DataValidator(uploaded_file)
    is_valid, errors, warnings = validator.validate()
    
    if not is_valid:
        return Response(
            {"error": "Validation failed", "details": errors},
            status=status.HTTP_400_BAD_REQUEST
        )
    
    # Create the data upload record
    data_upload = DataUpload.objects.create(
        file=uploaded_file,
        uploaded_by=request.user,
        status='pending'
    )
    
    # Log any warnings
    for warning in warnings:
        ProcessingLog.objects.create(
            data_upload=data_upload,
            message=warning,
            level='warning'
        )
    
    # Start the processing task
    process_csv_upload.delay(data_upload.id)
    
    return Response(
        {
            "message": "File uploaded successfully and is being processed.",
            "upload_id": data_upload.id,
            "warnings": warnings
        },
        status=status.HTTP_202_ACCEPTED
    )


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def upload_status_api(request, upload_id):
    """
    API endpoint to check the status of a data upload.
    """
    try:
        upload = DataUpload.objects.get(pk=upload_id, uploaded_by=request.user)
    except DataUpload.DoesNotExist:
        return Response(
            {"error": "Upload not found."},
            status=status.HTTP_404_NOT_FOUND
        )
    
    logs = upload.logs.all().order_by('-timestamp').values('timestamp', 'message', 'level')
    
    return Response({
        'id': upload.id,
        'file': upload.file.name,
        'uploaded_at': upload.uploaded_at,
        'status': upload.status,
        'rows_processed': upload.rows_processed,
        'rows_failed': upload.rows_failed,
        'error_message': upload.error_message,
        'logs': logs
    })


# Document Upload Views
@login_required
def upload_document(request):
    """
    View to handle document upload form.
    """
    if request.method == 'POST':
        if 'file' not in request.FILES:
            messages.error(request, "No file was uploaded.")
            return redirect('data_ingestion:upload_document')
        
        uploaded_file = request.FILES['file']
        file_name = uploaded_file.name.lower()
        
        # Check file extension
        if (not file_name.endswith('.pdf') and 
            not file_name.endswith('.docx') and 
            not file_name.endswith('.txt')):
            messages.error(request, "Only PDF, DOCX, and TXT files are accepted.")
            return redirect('data_ingestion:upload_document')
        
        # Determine document type
        if file_name.endswith('.pdf'):
            doc_type = 'pdf'
        elif file_name.endswith('.docx'):
            doc_type = 'docx'
        elif file_name.endswith('.txt'):
            doc_type = 'txt'
        else:
            doc_type = 'other'
        
        # Get student and course if provided
        student_id = request.POST.get('student_id')
        course_id = request.POST.get('course_id')
        assessment_id = request.POST.get('assessment_id')
        
        student = None
        course = None
        assessment = None
        
        if student_id:
            try:
                student = Student.objects.get(student_id=student_id)
            except Student.DoesNotExist:
                messages.warning(request, f"Student with ID {student_id} not found.")
        
        if course_id:
            try:
                course = Course.objects.get(course_id=course_id)
            except Course.DoesNotExist:
                messages.warning(request, f"Course with ID {course_id} not found.")
        
        if assessment_id:
            try:
                assessment = Assessment.objects.get(assessment_id=assessment_id)
            except Assessment.DoesNotExist:
                messages.warning(request, f"Assessment with ID {assessment_id} not found.")
        
        # Create the document upload record
        document = DocumentUpload.objects.create(
            file=uploaded_file,
            document_type=doc_type,
            uploaded_by=request.user,
            status='pending',
            student=student,
            course=course,
            assessment=assessment
        )
        
        # Start the processing task
        process_document_upload.delay(document.id)
        
        messages.success(
            request,
            f"Document '{uploaded_file.name}' has been uploaded and is being processed. "
            f"You can check the status at the document detail page."
        )

        return redirect('data_ingestion:document_detail', document_id=document.id)
    
    # Get students and courses for the form
    students = Student.objects.all()
    courses = Course.objects.all()
    assessments = Assessment.objects.all()
    
    return render(request, 'data_ingestion/upload_document.html', {
        'students': students,
        'courses': courses,
        'assessments': assessments
    })


@login_required
def document_list(request):
    """
    View to show a list of all document uploads.
    """
    documents = DocumentUpload.objects.filter(uploaded_by=request.user).order_by('-uploaded_at')
    return render(request, 'data_ingestion/document_list.html', {'documents': documents})


@login_required
def document_detail(request, document_id):
    """
    View to show details of a document upload, including extracted questions.
    """
    document = get_object_or_404(DocumentUpload, pk=document_id, uploaded_by=request.user)
    questions = document.questions.all().order_by('question_number')
    logs = ProcessingLog.objects.filter(document_upload=document).order_by('-timestamp')
    
    return render(request, 'data_ingestion/document_detail.html', {
        'document': document,
        'questions': questions,
        'logs': logs
    })


@require_POST
@login_required
def check_document_status(request, document_id):
    """
    AJAX endpoint to check the status of a document upload.
    """
    document = get_object_or_404(DocumentUpload, pk=document_id, uploaded_by=request.user)
    
    return JsonResponse({
        'status': document.status,
        'questions_processed': document.questions_processed,
        'questions_failed': document.questions_failed,
        'error_message': document.error_message
    })


# Document Upload API Views
@api_view(['POST'])
@permission_classes([IsAuthenticated])
def upload_document_api(request):
    """
    API endpoint to upload a document.
    """
    if 'file' not in request.FILES:
        return Response(
            {"error": "No file was uploaded."},
            status=status.HTTP_400_BAD_REQUEST
        )
    
    uploaded_file = request.FILES['file']
    file_name = uploaded_file.name.lower()
    
    # Check file extension
    if (not file_name.endswith('.pdf') and 
        not file_name.endswith('.docx') and 
        not file_name.endswith('.txt')):
        return Response(
            {"error": "Only PDF, DOCX, and TXT files are accepted."},
            status=status.HTTP_400_BAD_REQUEST
        )
    
    # Determine document type
    if file_name.endswith('.pdf'):
        doc_type = 'pdf'
    elif file_name.endswith('.docx'):
        doc_type = 'docx'
    elif file_name.endswith('.txt'):
        doc_type = 'txt'
    else:
        doc_type = 'other'
    
    # Get student and course if provided
    student_id = request.data.get('student_id')
    course_id = request.data.get('course_id')
    assessment_id = request.data.get('assessment_id')
    
    student = None
    course = None
    assessment = None
    
    if student_id:
        try:
            student = Student.objects.get(student_id=student_id)
        except Student.DoesNotExist:
            return Response(
                {"warning": f"Student with ID {student_id} not found."},
                status=status.HTTP_400_BAD_REQUEST
            )
    
    if course_id:
        try:
            course = Course.objects.get(course_id=course_id)
        except Course.DoesNotExist:
            return Response(
                {"warning": f"Course with ID {course_id} not found."},
                status=status.HTTP_400_BAD_REQUEST
            )
    
    if assessment_id:
        try:
            assessment = Assessment.objects.get(assessment_id=assessment_id)
        except Assessment.DoesNotExist:
            return Response(
                {"warning": f"Assessment with ID {assessment_id} not found."},
                status=status.HTTP_400_BAD_REQUEST
            )
    
    # Create the document upload record
    document = DocumentUpload.objects.create(
        file=uploaded_file,
        document_type=doc_type,
        uploaded_by=request.user,
        status='pending',
        student=student,
        course=course,
        assessment=assessment
    )
    
    # Start the processing task
    process_document_upload.delay(document.id)
    
    return Response(
        {
            "message": "Document uploaded successfully and is being processed.",
            "document_id": document.id
        },
        status=status.HTTP_202_ACCEPTED
    )


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def document_status_api(request, document_id):
    """
    API endpoint to check the status of a document upload.
    """
    try:
        document = DocumentUpload.objects.get(pk=document_id, uploaded_by=request.user)
    except DocumentUpload.DoesNotExist:
        return Response(
            {"error": "Document not found."},
            status=status.HTTP_404_NOT_FOUND
        )
    
    logs = ProcessingLog.objects.filter(document_upload=document).order_by('-timestamp').values('timestamp', 'message', 'level')
    questions = document.questions.all().values('id', 'question_number', 'page_number', 'is_processed', 'is_correct', 'confidence_score')
    
    return Response({
        'id': document.id,
        'file': document.file.name,
        'document_type': document.document_type,
        'uploaded_at': document.uploaded_at,
        'status': document.status,
        'questions_processed': document.questions_processed,
        'questions_failed': document.questions_failed,
        'error_message': document.error_message,
        'questions': list(questions),
        'logs': list(logs)
    })


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def extracted_question_api(request, question_id):
    """
    API endpoint to get details of an extracted question.
    """
    try:
        question = ExtractedQuestion.objects.get(pk=question_id)
        # Check if the user has permission to view this question
        if question.document.uploaded_by != request.user:
            return Response(
                {"error": "Permission denied."},
                status=status.HTTP_403_FORBIDDEN
            )
    except ExtractedQuestion.DoesNotExist:
        return Response(
            {"error": "Question not found."},
            status=status.HTTP_404_NOT_FOUND
        )
    
    return Response({
        'id': question.id,
        'document_id': question.document.id,
        'question_number': question.question_number,
        'page_number': question.page_number,
        'question_text': question.question_text,
        'student_answer': question.student_answer,
        'is_processed': question.is_processed,
        'topic': question.topic.name if question.topic else None,
        'is_correct': question.is_correct,
        'confidence_score': question.confidence_score,
        'feedback': question.feedback
    })
