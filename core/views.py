from django.shortcuts import render, redirect, get_object_or_404
from django.http import HttpResponse

def home(request):
    return render(request, 'core/home.html')

from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.http import JsonResponse
from django.views.decorators.http import require_POST
from django.db.models import Count, Avg

from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework import status

from .models import Student, Course, Topic, Resource, Assessment, Question, StudentInteraction


@login_required
def dashboard(request):
    """
    Main dashboard view for the PAL system.
    """
    # Count statistics for dashboard
    student_count = Student.objects.count()
    course_count = Course.objects.count()
    topic_count = Topic.objects.count()
    resource_count = Resource.objects.count()
    assessment_count = Assessment.objects.count()
    question_count = Question.objects.count()
    interaction_count = StudentInteraction.objects.count()
    
    # Recent activities
    recent_students = Student.objects.all().order_by('-id')[:5]
    recent_interactions = StudentInteraction.objects.select_related(
        'student', 'question', 'question__topic'
    ).order_by('-timestamp')[:10]
    
    # Courses with student counts
    popular_courses = Course.objects.annotate(
        student_count=Count('students')
    ).order_by('-student_count')[:5]
    
    # Topics with question counts
    popular_topics = Topic.objects.annotate(
        question_count=Count('questions')
    ).order_by('-question_count')[:5]
    
    return render(request, 'core/dashboard.html', {
        'student_count': student_count,
        'course_count': course_count,
        'topic_count': topic_count,
        'resource_count': resource_count,
        'assessment_count': assessment_count,
        'question_count': question_count,
        'interaction_count': interaction_count,
        'recent_students': recent_students,
        'recent_interactions': recent_interactions,
        'popular_courses': popular_courses,
        'popular_topics': popular_topics
    })


# Student Views
@login_required
def student_list(request):
    """
    View to show all students.
    """
    students = Student.objects.all().order_by('student_id')
    return render(request, 'core/student_list.html', {'students': students})


@login_required
def student_detail(request, student_id):
    """
    View to show details of a student.
    """
    student = get_object_or_404(Student, student_id=student_id)
    
    # Get courses for this student
    courses = student.courses.all()
    
    # Get recent interactions
    recent_interactions = StudentInteraction.objects.filter(
        student=student
    ).select_related('question', 'question__topic').order_by('-timestamp')[:20]
    
    # Calculate performance metrics per topic
    topic_performance = {}
    
    for interaction in StudentInteraction.objects.filter(
        student=student,
        question__topic__isnull=False
    ).select_related('question__topic'):
        topic = interaction.question.topic
        
        if topic.id not in topic_performance:
            topic_performance[topic.id] = {
                'topic': topic,
                'total': 0,
                'correct': 0,
                'percentage': 0
            }
        
        topic_performance[topic.id]['total'] += 1
        if interaction.correct:
            topic_performance[topic.id]['correct'] += 1
    
    # Calculate percentages
    for perf in topic_performance.values():
        if perf['total'] > 0:
            perf['percentage'] = (perf['correct'] / perf['total']) * 100
    
    # Sort by percentage
    topic_performance = sorted(
        topic_performance.values(),
        key=lambda x: x['percentage'],
        reverse=True
    )
    
    return render(request, 'core/student_detail.html', {
        'student': student,
        'courses': courses,
        'recent_interactions': recent_interactions,
        'topic_performance': topic_performance
    })


@login_required
def create_student(request):
    """
    View to create a new student.
    """
    if request.method == 'POST':
        student_id = request.POST.get('student_id')
        major = request.POST.get('major')
        academic_level = request.POST.get('academic_level')
        gpa = request.POST.get('gpa')
        study_frequency = request.POST.get('study_frequency')
        attendance_rate = request.POST.get('attendance_rate')
        participation_score = request.POST.get('participation_score')
        
        if not student_id:
            messages.error(request, "Student ID is required.")
            return redirect('create_student')
        
        # Check if student already exists
        if Student.objects.filter(student_id=student_id).exists():
            messages.error(request, f"Student with ID '{student_id}' already exists.")
            return redirect('create_student')
        
        # Create the student
        student = Student.objects.create(
            student_id=student_id,
            major=major,
            academic_level=academic_level,
            gpa=float(gpa) if gpa else 0.0,
            study_frequency=study_frequency,
            attendance_rate=float(attendance_rate) if attendance_rate else 0.0,
            participation_score=float(participation_score) if participation_score else 0.0
        )
        
        # Add to courses if specified
        course_ids = request.POST.getlist('courses')
        if course_ids:
            courses = Course.objects.filter(course_id__in=course_ids)
            for course in courses:
                student.courses.add(course)
        
        messages.success(request, f"Student '{student_id}' created successfully.")
        return redirect('student_detail', student_id=student.student_id)
    
    # GET request
    courses = Course.objects.all()
    return render(request, 'core/create_student.html', {
        'courses': courses,
        'academic_levels': Student.ACADEMIC_LEVEL_CHOICES,
        'study_frequencies': Student.STUDY_FREQUENCY_CHOICES
    })


@login_required
def edit_student(request, student_id):
    """
    View to edit a student.
    """
    student = get_object_or_404(Student, student_id=student_id)
    
    if request.method == 'POST':
        major = request.POST.get('major')
        academic_level = request.POST.get('academic_level')
        gpa = request.POST.get('gpa')
        study_frequency = request.POST.get('study_frequency')
        attendance_rate = request.POST.get('attendance_rate')
        participation_score = request.POST.get('participation_score')
        
        # Update the student
        student.major = major
        student.academic_level = academic_level
        student.gpa = float(gpa) if gpa else student.gpa
        student.study_frequency = study_frequency
        student.attendance_rate = float(attendance_rate) if attendance_rate else student.attendance_rate
        student.participation_score = float(participation_score) if participation_score else student.participation_score
        student.save()
        
        # Update courses
        course_ids = request.POST.getlist('courses')
        current_courses = student.courses.all()
        
        # Remove from courses not in the list
        for course in current_courses:
            if course.course_id not in course_ids:
                student.courses.remove(course)
        
        # Add to new courses
        for course_id in course_ids:
            if not student.courses.filter(course_id=course_id).exists():
                try:
                    course = Course.objects.get(course_id=course_id)
                    student.courses.add(course)
                except Course.DoesNotExist:
                    pass
        
        messages.success(request, f"Student '{student_id}' updated successfully.")
        return redirect('student_detail', student_id=student.student_id)
    
    # GET request
    courses = Course.objects.all()
    student_courses = student.courses.all()
    
    return render(request, 'core/edit_student.html', {
        'student': student,
        'courses': courses,
        'student_courses': student_courses,
        'academic_levels': Student.ACADEMIC_LEVEL_CHOICES,
        'study_frequencies': Student.STUDY_FREQUENCY_CHOICES
    })


# Course Views
@login_required
def course_list(request):
    """
    View to show all courses.
    """
    courses = Course.objects.all().order_by('course_id')
    
    # Add student count to each course
    for course in courses:
        course.student_count = course.students.count()
        course.topic_count = course.topics.count()
    
    return render(request, 'core/course_list.html', {'courses': courses})


@login_required
def course_detail(request, course_id):
    """
    View to show details of a course.
    """
    course = get_object_or_404(Course, course_id=course_id)
    
    # Get students in this course
    students = course.students.all()
    
    # Get topics for this course
    topics = course.topics.all()
    
    # Get assessments for this course
    assessments = course.assessments.all().order_by('-date')
    
    # Calculate average student performance
    course_performance = StudentInteraction.objects.filter(
        question__assessment__course=course
    ).aggregate(
        avg_score=Avg('score'),
        avg_correct=Avg('correct')
    )
    
    return render(request, 'core/course_detail.html', {
        'course': course,
        'students': students,
        'topics': topics,
        'assessments': assessments,
        'performance': course_performance,
        'student_count': students.count(),
        'topic_count': topics.count(),
        'assessment_count': assessments.count()
    })


@login_required
def create_course(request):
    """
    View to create a new course.
    """
    if request.method == 'POST':
        course_id = request.POST.get('course_id')
        title = request.POST.get('title')
        description = request.POST.get('description', '')
        
        if not course_id or not title:
            messages.error(request, "Course ID and title are required.")
            return redirect('create_course')
        
        # Check if course already exists
        if Course.objects.filter(course_id=course_id).exists():
            messages.error(request, f"Course with ID '{course_id}' already exists.")
            return redirect('create_course')
        
        # Create the course
        course = Course.objects.create(
            course_id=course_id,
            title=title,
            description=description
        )
        
        messages.success(request, f"Course '{title}' created successfully.")
        return redirect('course_detail', course_id=course.course_id)
    
    return render(request, 'core/create_course.html')


@login_required
def edit_course(request, course_id):
    """
    View to edit a course.
    """
    course = get_object_or_404(Course, course_id=course_id)
    
    if request.method == 'POST':
        title = request.POST.get('title')
        description = request.POST.get('description', '')
        
        if not title:
            messages.error(request, "Course title is required.")
            return redirect('edit_course', course_id=course_id)
        
        # Update the course
        course.title = title
        course.description = description
        course.save()
        
        messages.success(request, f"Course '{title}' updated successfully.")
        return redirect('course_detail', course_id=course.course_id)
    
    return render(request, 'core/edit_course.html', {'course': course})


@login_required
def course_students(request, course_id):
    """
    View to manage students in a course.
    """
    course = get_object_or_404(Course, course_id=course_id)
    
    if request.method == 'POST':
        action = request.POST.get('action')
        
        if action == 'add':
            student_ids = request.POST.getlist('students')
            for student_id in student_ids:
                try:
                    student = Student.objects.get(student_id=student_id)
                    course.students.add(student)
                except Student.DoesNotExist:
                    pass
            
            messages.success(request, f"Added {len(student_ids)} students to the course.")
        
        elif action == 'remove':
            student_ids = request.POST.getlist('selected_students')
            for student_id in student_ids:
                try:
                    student = Student.objects.get(student_id=student_id)
                    course.students.remove(student)
                except Student.DoesNotExist:
                    pass
            
            messages.success(request, f"Removed {len(student_ids)} students from the course.")
        
        return redirect('course_students', course_id=course_id)
    
    # GET request
    enrolled_students = course.students.all().order_by('student_id')
    available_students = Student.objects.exclude(
        courses=course
    ).order_by('student_id')
    
    return render(request, 'core/course_students.html', {
        'course': course,
        'enrolled_students': enrolled_students,
        'available_students': available_students
    })


# Topic Views
@login_required
def topic_list(request):
    """
    View to show all topics.
    """
    topics = Topic.objects.all().select_related('course', 'parent')
    
    # Filter by course if specified
    course_id = request.GET.get('course_id')
    if course_id:
        topics = topics.filter(course__course_id=course_id)
    
    # Group by course
    topics_by_course = {}
    for topic in topics:
        if topic.course not in topics_by_course:
            topics_by_course[topic.course] = []
        topics_by_course[topic.course].append(topic)
    
    # Get all courses for filter
    courses = Course.objects.all()
    
    return render(request, 'core/topic_list.html', {
        'topics_by_course': topics_by_course,
        'courses': courses,
        'selected_course': course_id
    })


@login_required
def topic_detail(request, topic_id):
    """
    View to show details of a topic.
    """
    topic = get_object_or_404(Topic, pk=topic_id)
    
    # Get parent and subtopics
    parent = topic.parent
    subtopics = Topic.objects.filter(parent=topic)
    
    # Get resources for this topic
    resources = topic.resources.all()
    
    # Get questions for this topic
    questions = topic.questions.all()
    
    # Calculate student performance for this topic
    topic_performance = StudentInteraction.objects.filter(
        question__topic=topic
    ).aggregate(
        avg_score=Avg('score'),
        avg_correct=Avg('correct')
    )
    
    return render(request, 'core/topic_detail.html', {
        'topic': topic,
        'parent': parent,
        'subtopics': subtopics,
        'resources': resources,
        'questions': questions,
        'performance': topic_performance
    })


@login_required
def create_topic(request):
    """
    View to create a new topic.
    """
    if request.method == 'POST':
        name = request.POST.get('name')
        description = request.POST.get('description', '')
        course_id = request.POST.get('course')
        parent_id = request.POST.get('parent')
        
        if not name or not course_id:
            messages.error(request, "Topic name and course are required.")
            return redirect('create_topic')
        
        try:
            course = Course.objects.get(pk=course_id)
        except Course.DoesNotExist:
            messages.error(request, "Selected course does not exist.")
            return redirect('create_topic')
        
        # Get parent topic if specified
        parent = None
        if parent_id:
            try:
                parent = Topic.objects.get(pk=parent_id)
                # Ensure parent is from the same course
                if parent.course != course:
                    messages.error(request, "Parent topic must be from the same course.")
                    return redirect('create_topic')
            except Topic.DoesNotExist:
                pass
        
        # Create the topic
        topic = Topic.objects.create(
            name=name,
            description=description,
            course=course,
            parent=parent
        )
        
        messages.success(request, f"Topic '{name}' created successfully.")
        return redirect('topic_detail', topic_id=topic.id)
    
    # GET request
    courses = Course.objects.all()
    
    # Get topics for parent selection
    course_id = request.GET.get('course_id')
    topics = []
    if course_id:
        topics = Topic.objects.filter(course__pk=course_id)
    
    return render(request, 'core/create_topic.html', {
        'courses': courses,
        'topics': topics,
        'selected_course': course_id
    })


@login_required
def edit_topic(request, topic_id):
    """
    View to edit a topic.
    """
    topic = get_object_or_404(Topic, pk=topic_id)
    
    if request.method == 'POST':
        name = request.POST.get('name')
        description = request.POST.get('description', '')
        parent_id = request.POST.get('parent')
        
        if not name:
            messages.error(request, "Topic name is required.")
            return redirect('edit_topic', topic_id=topic_id)
        
        # Get parent topic if specified
        parent = None
        if parent_id:
            try:
                parent = Topic.objects.get(pk=parent_id)
                # Ensure parent is from the same course and not a circular reference
                if parent.course != topic.course or parent.id == topic.id:
                    messages.error(request, "Invalid parent topic selection.")
                    return redirect('edit_topic', topic_id=topic_id)
            except Topic.DoesNotExist:
                pass
        
        # Update the topic
        topic.name = name
        topic.description = description
        topic.parent = parent
        topic.save()
        
        messages.success(request, f"Topic '{name}' updated successfully.")
        return redirect('topic_detail', topic_id=topic.id)
    
    # GET request
    potential_parents = Topic.objects.filter(
        course=topic.course
    ).exclude(
        pk=topic_id
    ).exclude(
        parent=topic
    )  # Avoid circular references
    
    return render(request, 'core/edit_topic.html', {
        'topic': topic,
        'potential_parents': potential_parents
    })


# Resource Views
@login_required
def resource_list(request):
    """
    View to show all resources.
    """
    resources = Resource.objects.all()
    
    # Filter by topic if specified
    topic_id = request.GET.get('topic_id')
    if topic_id:
        resources = resources.filter(topics__id=topic_id)
    
    # Filter by type if specified
    resource_type = request.GET.get('type')
    if resource_type:
        resources = resources.filter(resource_type=resource_type)
    
    # Get all topics and resource types for filters
    topics = Topic.objects.all()
    resource_types = Resource.TYPE_CHOICES
    
    return render(request, 'core/resource_list.html', {
        'resources': resources,
        'topics': topics,
        'resource_types': resource_types,
        'selected_topic': topic_id,
        'selected_type': resource_type
    })


@login_required
def resource_detail(request, resource_id):
    """
    View to show details of a resource.
    """
    resource = get_object_or_404(Resource, pk=resource_id)
    
    # Get topics for this resource
    topics = resource.topics.all()
    
    return render(request, 'core/resource_detail.html', {
        'resource': resource,
        'topics': topics
    })


@login_required
def create_resource(request):
    """
    View to create a new resource.
    """
    if request.method == 'POST':
        title = request.POST.get('title')
        description = request.POST.get('description', '')
        url = request.POST.get('url')
        resource_type = request.POST.get('resource_type')
        difficulty = request.POST.get('difficulty')
        estimated_hours = request.POST.get('estimated_hours')
        estimated_minutes = request.POST.get('estimated_minutes')
        
        if not title or not url or not resource_type or not difficulty:
            messages.error(request, "Title, URL, resource type, and difficulty are required.")
            return redirect('create_resource')
        
        # Create the resource
        import datetime
        estimated_time = datetime.timedelta(
            hours=int(estimated_hours) if estimated_hours else 0,
            minutes=int(estimated_minutes) if estimated_minutes else 0
        )
        
        resource = Resource.objects.create(
            title=title,
            description=description,
            url=url,
            resource_type=resource_type,
            difficulty=difficulty,
            estimated_time=estimated_time
        )
        
        # Add topics
        topic_ids = request.POST.getlist('topics')
        if topic_ids:
            topics = Topic.objects.filter(pk__in=topic_ids)
            for topic in topics:
                resource.topics.add(topic)
        
        messages.success(request, f"Resource '{title}' created successfully.")
        return redirect('resource_detail', resource_id=resource.id)
    
    # GET request
    topics = Topic.objects.all()
    return render(request, 'core/create_resource.html', {
        'topics': topics,
        'resource_types': Resource.TYPE_CHOICES,
        'difficulty_levels': Resource.DIFFICULTY_CHOICES
    })


@login_required
def edit_resource(request, resource_id):
    """
    View to edit a resource.
    """
    resource = get_object_or_404(Resource, pk=resource_id)
    
    if request.method == 'POST':
        title = request.POST.get('title')
        description = request.POST.get('description', '')
        url = request.POST.get('url')
        resource_type = request.POST.get('resource_type')
        difficulty = request.POST.get('difficulty')
        estimated_hours = request.POST.get('estimated_hours')
        estimated_minutes = request.POST.get('estimated_minutes')
        
        if not title or not url or not resource_type or not difficulty:
            messages.error(request, "Title, URL, resource type, and difficulty are required.")
            return redirect('edit_resource', resource_id=resource_id)
        
        # Update the resource
        import datetime
        estimated_time = datetime.timedelta(
            hours=int(estimated_hours) if estimated_hours else 0,
            minutes=int(estimated_minutes) if estimated_minutes else 0
        )
        
        resource.title = title
        resource.description = description
        resource.url = url
        resource.resource_type = resource_type
        resource.difficulty = difficulty
        resource.estimated_time = estimated_time
        resource.save()
        
        # Update topics
        topic_ids = request.POST.getlist('topics')
        resource.topics.clear()
        if topic_ids:
            topics = Topic.objects.filter(pk__in=topic_ids)
            for topic in topics:
                resource.topics.add(topic)
        
        messages.success(request, f"Resource '{title}' updated successfully.")
        return redirect('resource_detail', resource_id=resource.id)
    
    # GET request
    topics = Topic.objects.all()
    resource_topics = resource.topics.all()
    
    # Calculate hours and minutes for the template
    hours = resource.estimated_time.total_seconds() // 3600
    minutes = (resource.estimated_time.total_seconds() % 3600) // 60
    
    return render(request, 'core/edit_resource.html', {
        'resource': resource,
        'topics': topics,
        'resource_topics': resource_topics,
        'resource_types': Resource.TYPE_CHOICES,
        'difficulty_levels': Resource.DIFFICULTY_CHOICES,
        'hours': int(hours),
        'minutes': int(minutes)
    })


# Assessment Views
@login_required
def assessment_list(request):
    """
    View to show all assessments.
    """
    assessments = Assessment.objects.all().select_related('course')
    
    # Filter by course if specified
    course_id = request.GET.get('course_id')
    if course_id:
        assessments = assessments.filter(course__course_id=course_id)
    
    # Filter by type if specified
    assessment_type = request.GET.get('type')
    if assessment_type:
        assessments = assessments.filter(assessment_type=assessment_type)
    
    # Get all courses and assessment types for filters
    courses = Course.objects.all()
    assessment_types = Assessment.ASSESSMENT_TYPE_CHOICES
    
    return render(request, 'core/assessment_list.html', {
        'assessments': assessments,
        'courses': courses,
        'assessment_types': assessment_types,
        'selected_course': course_id,
        'selected_type': assessment_type
    })


@login_required
def assessment_detail(request, assessment_id):
    """
    View to show details of an assessment.
    """
    assessment = get_object_or_404(Assessment, assessment_id=assessment_id)
    
    # Get questions for this assessment
    questions = assessment.questions.all()
    
    # Calculate student performance
    assessment_performance = StudentInteraction.objects.filter(
        question__assessment=assessment
    ).aggregate(
        avg_score=Avg('score'),
        avg_correct=Avg('correct')
    )
    
    return render(request, 'core/assessment_detail.html', {
        'assessment': assessment,
        'questions': questions,
        'performance': assessment_performance,
        'question_count': questions.count()
    })


@login_required
def create_assessment(request):
    """
    View to create a new assessment.
    """
    if request.method == 'POST':
        assessment_id = request.POST.get('assessment_id')
        title = request.POST.get('title')
        assessment_type = request.POST.get('assessment_type')
        course_id = request.POST.get('course')
        date_str = request.POST.get('date')
        proctored = request.POST.get('proctored') == 'on'
        
        if not assessment_id or not title or not assessment_type or not course_id or not date_str:
            messages.error(request, "All fields are required.")
            return redirect('create_assessment')
        
        try:
            course = Course.objects.get(pk=course_id)
        except Course.DoesNotExist:
            messages.error(request, "Selected course does not exist.")
            return redirect('create_assessment')
        
        # Check if assessment already exists
        if Assessment.objects.filter(assessment_id=assessment_id).exists():
            messages.error(request, f"Assessment with ID '{assessment_id}' already exists.")
            return redirect('create_assessment')
        
        # Parse date
        from django.utils.dateparse import parse_datetime
        date = parse_datetime(date_str)
        if not date:
            messages.error(request, "Invalid date format.")
            return redirect('create_assessment')
        
        # Create the assessment
        assessment = Assessment.objects.create(
            assessment_id=assessment_id,
            title=title,
            assessment_type=assessment_type,
            course=course,
            date=date,
            proctored=proctored
        )
        
        messages.success(request, f"Assessment '{title}' created successfully.")
        return redirect('assessment_detail', assessment_id=assessment.assessment_id)
    
    # GET request
    courses = Course.objects.all()
    return render(request, 'core/create_assessment.html', {
        'courses': courses,
        'assessment_types': Assessment.ASSESSMENT_TYPE_CHOICES
    })


@login_required
def edit_assessment(request, assessment_id):
    """
    View to edit an assessment.
    """
    assessment = get_object_or_404(Assessment, assessment_id=assessment_id)
    
    if request.method == 'POST':
        title = request.POST.get('title')
        assessment_type = request.POST.get('assessment_type')
        date_str = request.POST.get('date')
        proctored = request.POST.get('proctored') == 'on'
        
        if not title or not assessment_type or not date_str:
            messages.error(request, "All fields are required.")
            return redirect('edit_assessment', assessment_id=assessment_id)
        
        # Parse date
        from django.utils.dateparse import parse_datetime
        date = parse_datetime(date_str)
        if not date:
            messages.error(request, "Invalid date format.")
            return redirect('edit_assessment', assessment_id=assessment_id)
        
        # Update the assessment
        assessment.title = title
        assessment.assessment_type = assessment_type
        assessment.date = date
        assessment.proctored = proctored
        assessment.save()
        
        messages.success(request, f"Assessment '{title}' updated successfully.")
        return redirect('assessment_detail', assessment_id=assessment.assessment_id)
    
    # GET request
    return render(request, 'core/edit_assessment.html', {
        'assessment': assessment,
        'assessment_types': Assessment.ASSESSMENT_TYPE_CHOICES
    })


# Question Views
@login_required
def question_list(request):
    """
    View to show all questions.
    """
    questions = Question.objects.all().select_related('assessment', 'topic')
    
    # Filter by assessment if specified
    assessment_id = request.GET.get('assessment_id')
    if assessment_id:
        questions = questions.filter(assessment__assessment_id=assessment_id)
    
    # Filter by topic if specified
    topic_id = request.GET.get('topic_id')
    if topic_id:
        questions = questions.filter(topic__id=topic_id)
    
    # Get all assessments and topics for filters
    assessments = Assessment.objects.all()
    topics = Topic.objects.all()
    
    return render(request, 'core/question_list.html', {
        'questions': questions,
        'assessments': assessments,
        'topics': topics,
        'selected_assessment': assessment_id,
        'selected_topic': topic_id
    })


@login_required
def question_detail(request, question_id):
    """
    View to show details of a question.
    """
    question = get_object_or_404(Question, question_id=question_id)
    
    # Get interactions for this question
    interactions = StudentInteraction.objects.filter(
        question=question
    ).select_related('student').order_by('-timestamp')
    
    # Calculate performance stats
    correct_count = interactions.filter(correct=True).count()
    total_count = interactions.count()
    
    if total_count > 0:
        correct_percentage = (correct_count / total_count) * 100
    else:
        correct_percentage = 0
    
    return render(request, 'core/question_detail.html', {
        'question': question,
        'interactions': interactions,
        'correct_count': correct_count,
        'total_count': total_count,
        'correct_percentage': correct_percentage
    })


@login_required
def create_question(request):
    """
    View to create a new question.
    """
    if request.method == 'POST':
        question_id = request.POST.get('question_id')
        assessment_id = request.POST.get('assessment')
        text = request.POST.get('text')
        question_type = request.POST.get('question_type')
        topic_id = request.POST.get('topic')
        
        if not question_id or not assessment_id or not text or not question_type:
            messages.error(request, "Question ID, assessment, text, and type are required.")
            return redirect('create_question')
        
        try:
            assessment = Assessment.objects.get(assessment_id=assessment_id)
        except Assessment.DoesNotExist:
            messages.error(request, "Selected assessment does not exist.")
            return redirect('create_question')
        
        # Check if question already exists
        if Question.objects.filter(question_id=question_id).exists():
            messages.error(request, f"Question with ID '{question_id}' already exists.")
            return redirect('create_question')
        
        # Get topic if specified
        topic = None
        if topic_id:
            try:
                topic = Topic.objects.get(pk=topic_id)
            except Topic.DoesNotExist:
                pass
        
        # Create the question
        question = Question.objects.create(
            question_id=question_id,
            assessment=assessment,
            text=text,
            question_type=question_type,
            topic=topic
        )
        
        messages.success(request, f"Question '{question_id}' created successfully.")
        return redirect('question_detail', question_id=question.question_id)
    
    # GET request
    assessments = Assessment.objects.all()
    topics = Topic.objects.all()
    
    # Filter topics by course if assessment is specified
    assessment_id = request.GET.get('assessment_id')
    if assessment_id:
        try:
            assessment = Assessment.objects.get(assessment_id=assessment_id)
            topics = topics.filter(course=assessment.course)
        except Assessment.DoesNotExist:
            pass
    
    return render(request, 'core/create_question.html', {
        'assessments': assessments,
        'topics': topics,
        'question_types': Question.QUESTION_TYPE_CHOICES,
        'selected_assessment': assessment_id
    })


@login_required
def edit_question(request, question_id):
    """
    View to edit a question.
    """
    question = get_object_or_404(Question, question_id=question_id)
    
    if request.method == 'POST':
        text = request.POST.get('text')
        question_type = request.POST.get('question_type')
        topic_id = request.POST.get('topic')
        
        if not text or not question_type:
            messages.error(request, "Question text and type are required.")
            return redirect('edit_question', question_id=question_id)
        
        # Get topic if specified
        topic = None
        if topic_id:
            try:
                topic = Topic.objects.get(pk=topic_id)
            except Topic.DoesNotExist:
                pass
        
        # Update the question
        question.text = text
        question.question_type = question_type
        question.topic = topic
        question.save()
        
        messages.success(request, f"Question '{question_id}' updated successfully.")
        return redirect('question_detail', question_id=question.question_id)
    
    # GET request
    topics = Topic.objects.filter(course=question.assessment.course)
    
    return render(request, 'core/edit_question.html', {
        'question': question,
        'topics': topics,
        'question_types': Question.QUESTION_TYPE_CHOICES
    })


# Student Interaction Views
@login_required
def interaction_list(request):
    """
    View to show all student interactions.
    """
    interactions = StudentInteraction.objects.all().select_related(
        'student', 'question', 'question__topic'
    ).order_by('-timestamp')
    
    # Filter by student if specified
    student_id = request.GET.get('student_id')
    if student_id:
        interactions = interactions.filter(student__student_id=student_id)
    
    # Filter by topic if specified
    topic_id = request.GET.get('topic_id')
    if topic_id:
        interactions = interactions.filter(question__topic__id=topic_id)
    
    # Filter by correctness if specified
    correctness = request.GET.get('correct')
    if correctness:
        correct = correctness == 'true'
        interactions = interactions.filter(correct=correct)
    
    # Paginate results
    from django.core.paginator import Paginator
    paginator = Paginator(interactions, 25)  # Show 25 interactions per page
    page = request.GET.get('page')
    interactions = paginator.get_page(page)
    
    # Get all students and topics for filters
    students = Student.objects.all()
    topics = Topic.objects.all()
    
    return render(request, 'core/interaction_list.html', {
        'interactions': interactions,
        'students': students,
        'topics': topics,
        'selected_student': student_id,
        'selected_topic': topic_id,
        'selected_correct': correctness
    })


@login_required
def create_interaction(request):
    """
    View to create a new student interaction.
    """
    if request.method == 'POST':
        student_id = request.POST.get('student')
        question_id = request.POST.get('question')
        response = request.POST.get('response', '')
        correct = request.POST.get('correct') == 'on'
        score = request.POST.get('score')
        time_minutes = request.POST.get('time_minutes')
        time_seconds = request.POST.get('time_seconds')
        timestamp = request.POST.get('timestamp')
        attempt_number = request.POST.get('attempt_number', 1)
        resource_viewed = request.POST.get('resource_viewed') == 'on'
        
        if not student_id or not question_id:
            messages.error(request, "Student and question are required.")
            return redirect('create_interaction')
        
        try:
            student = Student.objects.get(student_id=student_id)
        except Student.DoesNotExist:
            messages.error(request, "Selected student does not exist.")
            return redirect('create_interaction')
        
        try:
            question = Question.objects.get(question_id=question_id)
        except Question.DoesNotExist:
            messages.error(request, "Selected question does not exist.")
            return redirect('create_interaction')
        
        # Create time taken duration
        import datetime
        time_taken = datetime.timedelta(
            minutes=int(time_minutes) if time_minutes else 0,
            seconds=int(time_seconds) if time_seconds else 0
        )
        
        # Parse timestamp
        from django.utils.dateparse import parse_datetime
        from django.utils import timezone
        if timestamp:
            timestamp_dt = parse_datetime(timestamp)
        else:
            timestamp_dt = timezone.now()
        
        # Create the interaction
        interaction = StudentInteraction.objects.create(
            student=student,
            question=question,
            response=response,
            correct=correct,
            score=float(score) if score else None,
            time_taken=time_taken,
            timestamp=timestamp_dt,
            attempt_number=int(attempt_number),
            resource_viewed_before=resource_viewed
        )
        
        messages.success(request, "Student interaction created successfully.")
        return redirect('interaction_list')
    
    # GET request
    students = Student.objects.all()
    questions = Question.objects.all()
    
    return render(request, 'core/create_interaction.html', {
        'students': students,
        'questions': questions
    })


# API Views
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_students_api(request):
    """
    API endpoint to get all students.
    """
    course_id = request.query_params.get('course_id')
    
    if course_id:
        # Filter by course
        try:
            course = Course.objects.get(course_id=course_id)
            students = Student.objects.filter(courses=course)
        except Course.DoesNotExist:
            return Response(
                {"error": "Course not found."},
                status=status.HTTP_404_NOT_FOUND
            )
    else:
        students = Student.objects.all()
    
    result = []
    for student in students:
        result.append({
            "student_id": student.student_id,
            "major": student.major,
            "academic_level": student.academic_level,
            "gpa": student.gpa,
            "study_frequency": student.study_frequency,
            "attendance_rate": student.attendance_rate,
            "participation_score": student.participation_score
        })
    
    return Response(result)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_student_api(request, student_id):
    """
    API endpoint to get a specific student.
    """
    try:
        student = Student.objects.get(student_id=student_id)
    except Student.DoesNotExist:
        return Response(
            {"error": "Student not found."},
            status=status.HTTP_404_NOT_FOUND
        )
    
    # Get courses
    courses = []
    for course in student.courses.all():
        courses.append({
            "course_id": course.course_id,
            "title": course.title
        })
    
    return Response({
        "student_id": student.student_id,
        "major": student.major,
        "academic_level": student.academic_level,
        "gpa": student.gpa,
        "study_frequency": student.study_frequency,
        "attendance_rate": student.attendance_rate,
        "participation_score": student.participation_score,
        "courses": courses
    })


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_courses_api(request):
    """
    API endpoint to get all courses.
    """
    courses = Course.objects.all()
    
    result = []
    for course in courses:
        result.append({
            "course_id": course.course_id,
            "title": course.title,
            "description": course.description,
            "student_count": course.students.count()
        })
    
    return Response(result)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_course_api(request, course_id):
    """
    API endpoint to get a specific course.
    """
    try:
        course = Course.objects.get(course_id=course_id)
    except Course.DoesNotExist:
        return Response(
            {"error": "Course not found."},
            status=status.HTTP_404_NOT_FOUND
        )
    
    # Get topics
    topics = []
    for topic in Topic.objects.filter(course=course, parent__isnull=True):
        subtopics = []
        for subtopic in Topic.objects.filter(parent=topic):
            subtopics.append({
                "id": subtopic.id,
                "name": subtopic.name
            })
        
        topics.append({
            "id": topic.id,
            "name": topic.name,
            "description": topic.description,
            "subtopics": subtopics
        })
    
    # Get assessments
    assessments = []
    for assessment in Assessment.objects.filter(course=course):
        assessments.append({
            "assessment_id": assessment.assessment_id,
            "title": assessment.title,
            "assessment_type": assessment.assessment_type,
            "date": assessment.date
        })
    
    # Get student count
    student_count = course.students.count()
    
    return Response({
        "course_id": course.course_id,
        "title": course.title,
        "description": course.description,
        "topics": topics,
        "assessments": assessments,
        "student_count": student_count
    })


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_topics_api(request):
    """
    API endpoint to get all topics.
    """
    course_id = request.query_params.get('course_id')
    
    if course_id:
        # Filter by course
        try:
            course = Course.objects.get(course_id=course_id)
            topics = Topic.objects.filter(course=course)
        except Course.DoesNotExist:
            return Response(
                {"error": "Course not found."},
                status=status.HTTP_404_NOT_FOUND
            )
    else:
        topics = Topic.objects.all()
    
    result = []
    for topic in topics:
        result.append({
            "id": topic.id,
            "name": topic.name,
            "description": topic.description,
            "course_id": topic.course.course_id,
            "parent_id": topic.parent_id
        })
    
    return Response(result)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_topic_api(request, topic_id):
    """
    API endpoint to get a specific topic.
    """
    try:
        topic = Topic.objects.get(pk=topic_id)
    except Topic.DoesNotExist:
        return Response(
            {"error": "Topic not found."},
            status=status.HTTP_404_NOT_FOUND
        )
    
    # Get parent
    parent = None
    if topic.parent:
        parent = {
            "id": topic.parent.id,
            "name": topic.parent.name
        }
    
    # Get subtopics
    subtopics = []
    for subtopic in Topic.objects.filter(parent=topic):
        subtopics.append({
            "id": subtopic.id,
            "name": subtopic.name
        })
    
    # Get resources
    resources = []
    for resource in topic.resources.all():
        resources.append({
            "id": resource.id,
            "title": resource.title,
            "url": resource.url,
            "resource_type": resource.resource_type,
            "difficulty": resource.difficulty
        })
    
    return Response({
        "id": topic.id,
        "name": topic.name,
        "description": topic.description,
        "course_id": topic.course.course_id,
        "parent": parent,
        "subtopics": subtopics,
        "resources": resources
    })


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_resources_api(request):
    """
    API endpoint to get all resources.
    """
    topic_id = request.query_params.get('topic_id')
    
    if topic_id:
        # Filter by topic
        try:
            topic = Topic.objects.get(pk=topic_id)
            resources = Resource.objects.filter(topics=topic)
        except Topic.DoesNotExist:
            return Response(
                {"error": "Topic not found."},
                status=status.HTTP_404_NOT_FOUND
            )
    else:
        resources = Resource.objects.all()
    
    result = []
    for resource in resources:
        result.append({
            "id": resource.id,
            "title": resource.title,
            "url": resource.url,
            "resource_type": resource.resource_type,
            "difficulty": resource.difficulty,
            "estimated_time": str(resource.estimated_time)
        })
    
    return Response(result)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_resource_api(request, resource_id):
    """
    API endpoint to get a specific resource.
    """
    try:
        resource = Resource.objects.get(pk=resource_id)
    except Resource.DoesNotExist:
        return Response(
            {"error": "Resource not found."},
            status=status.HTTP_404_NOT_FOUND
        )
    
    # Get topics
    topics = []
    for topic in resource.topics.all():
        topics.append({
            "id": topic.id,
            "name": topic.name,
            "course_id": topic.course.course_id
        })
    
    return Response({
        "id": resource.id,
        "title": resource.title,
        "description": resource.description,
        "url": resource.url,
        "resource_type": resource.resource_type,
        "difficulty": resource.difficulty,
        "estimated_time": str(resource.estimated_time),
        "topics": topics
    })


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_student_performance_api(request, student_id):
    """
    API endpoint to get a student's performance across topics.
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
    
    if course_id:
        try:
            course = Course.objects.get(course_id=course_id)
            topics = Topic.objects.filter(course=course)
        except Course.DoesNotExist:
            return Response(
                {"error": "Course not found."},
                status=status.HTTP_404_NOT_FOUND
            )
    else:
        # Get all topics for courses this student is enrolled in
        topics = Topic.objects.filter(course__in=student.courses.all())
    
    # Calculate performance for each topic
    topic_performance = []
    
    for topic in topics:
        # Get interactions for this topic
        interactions = StudentInteraction.objects.filter(
            student=student,
            question__topic=topic
        )
        
        total = interactions.count()
        correct = interactions.filter(correct=True).count()
        
        if total > 0:
            accuracy = correct / total
        else:
            accuracy = 0
        
        topic_performance.append({
            "topic_id": topic.id,
            "topic_name": topic.name,
            "total_questions": total,
            "correct_answers": correct,
            "accuracy": accuracy
        })
    
    return Response({
        "student_id": student.student_id,
        "topic_performance": topic_performance
    })
