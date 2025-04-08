from django.test import TestCase
from django.urls import reverse
from django.contrib.auth.models import User
from rest_framework.test import APIClient
from rest_framework import status
from django.utils import timezone
import datetime

from core.models import Student, Course, Topic
from knowledge_graph.models import KnowledgeGraph
from ml_models.models import KnowledgeTracingModel, PredictionBatch, TopicMastery
from learning_paths.models import PathGenerator, PathGenerationJob, LearningPath, LearningPathItem


class APIEndpointTests(TestCase):
    """Test class for API endpoints."""
    
    def setUp(self):
        """Set up test data."""
        # Create a user
        self.user = User.objects.create_user(
            username='testuser',
            password='testpassword'
        )
        
        # Create a test student
        self.student = Student.objects.create(
            student_id='S12345',
            major='Computer Science',
            academic_level='freshman',
            gpa=3.5,
            study_frequency='daily',
            attendance_rate=90.0,
            participation_score=85.0
        )
        
        # Create a test course
        self.course = Course.objects.create(
            course_id='CS101',
            title='Introduction to Java Programming',
            description='A beginner course on Java programming'
        )
        
        # Add student to course
        self.course.students.add(self.student)
        
        # Create topics
        self.topic1 = Topic.objects.create(
            name='Variables',
            description='Variables in Java',
            course=self.course
        )
        
        self.topic2 = Topic.objects.create(
            name='Loops',
            description='Loops in Java',
            course=self.course
        )
        
        # Create a knowledge graph
        self.graph = KnowledgeGraph.objects.create(
            name='Java KSG',
            description='Knowledge Structure Graph for Java',
            created_by=self.user,
            version='1.0',
            data={
                'nodes': [
                    {'id': self.topic1.id, 'name': 'Variables'},
                    {'id': self.topic2.id, 'name': 'Loops'}
                ],
                'edges': [
                    {
                        'source': self.topic1.id,
                        'target': self.topic2.id,
                        'relationship': 'prerequisite',
                        'weight': 1.0
                    }
                ]
            },
            is_active=True
        )
        
        # Create a model
        self.model = KnowledgeTracingModel.objects.create(
            name='Test DKT Model',
            model_type='dkt',
            description='Test model',
            created_by=self.user,
            model_path='/test/path/model.pt',
            status='active',
            is_default=True,
            course=self.course,
            hyperparameters={
                'hidden_size': 100,
                'num_layers': 1
            }
        )
        
        # Create a prediction batch
        self.batch = PredictionBatch.objects.create(
            model=self.model,
            status='completed',
            total_students=1,
            processed_students=1,
            completed_at=timezone.now()
        )
        
        # Create masteries
        self.mastery1 = TopicMastery.objects.create(
            student=self.student,
            topic=self.topic1,
            prediction_batch=self.batch,
            mastery_score=0.8,
            confidence=0.9,
            trend='improving'
        )
        
        self.mastery2 = TopicMastery.objects.create(
            student=self.student,
            topic=self.topic2,
            prediction_batch=self.batch,
            mastery_score=0.4,
            confidence=0.7,
            trend='stagnant'
        )
        
        # Create a path generator
        self.generator = PathGenerator.objects.create(
            name='Default Generator',
            description='Default path generator',
            created_by=self.user,
            is_active=True,
            config={
                'include_strong_topics': False,
                'max_resources_per_topic': 3
            }
        )
        
        # Create a learning path job
        self.job = PathGenerationJob.objects.create(
            generator=self.generator,
            student=self.student,
            course=self.course,
            status='completed',
            prediction_batch=self.batch,
            knowledge_graph=self.graph,
            completed_at=timezone.now()
        )
        
        # Create a learning path
        self.path = LearningPath.objects.create(
            generation_job=self.job,
            student=self.student,
            course=self.course,
            name='Learning Path for CS101',
            description='Generated learning path',
            generated_at=timezone.now(),
            status='active',
            overall_progress={
                'completed_topics': 0,
                'in_progress_topics': 0,
                'not_started_topics': 2,
                'overall_mastery': 0.6
            },
            estimated_completion_time=datetime.timedelta(hours=2)
        )
        
        # Create path items
        self.path_item1 = LearningPathItem.objects.create(
            path=self.path,
            topic=self.topic2,  # Loops (lower mastery)
            priority=1,
            status='developing',
            proficiency_score=0.4,
            trend='stagnant',
            confidence_of_improvement=0.6,
            reason='You need to improve your understanding of loops',
            estimated_review_time=datetime.timedelta(minutes=60),
            completed=False
        )
        
        self.path_item2 = LearningPathItem.objects.create(
            path=self.path,
            topic=self.topic1,  # Variables (higher mastery)
            priority=2,
            status='strong',
            proficiency_score=0.8,
            trend='improving',
            confidence_of_improvement=0.9,
            reason='You have a good understanding of variables',
            estimated_review_time=datetime.timedelta(minutes=30),
            completed=False
        )
        
        # Set up API client
        self.client = APIClient()
        self.client.force_authenticate(user=self.user)
    
    def test_student_profile_api(self):
        """Test the student profile API endpoint."""
        url = reverse('student_profile_api', args=[self.student.student_id])
        response = self.client.get(url, {'course_id': self.course.course_id})
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['student']['id'], self.student.student_id)
        self.assertEqual(response.data['course']['id'], self.course.course_id)
        self.assertEqual(len(response.data['knowledge_states']), 2)
        self.assertIsNotNone(response.data['learning_path'])
    
    def test_generate_recommendations_api(self):
        """Test the generate recommendations API endpoint."""
        url = reverse('generate_recommendations_api', args=[self.student.student_id])
        response = self.client.post(url, {'course_id': self.course.course_id})
        
        self.assertEqual(response.status_code, status.HTTP_202_ACCEPTED)
        self.assertIn('prediction_batch_id', response.data)
        self.assertIn('job_id', response.data)
    
    def test_check_recommendation_status_api(self):
        """Test the check recommendation status API endpoint."""
        # Create a pending job and batch to check status
        batch = PredictionBatch.objects.create(
            model=self.model,
            status='pending'
        )
        
        job = PathGenerationJob.objects.create(
            generator=self.generator,
            student=self.student,
            course=self.course,
            status='pending',
            prediction_batch=batch
        )
        
        url = reverse('check_recommendation_status_api')
        response = self.client.get(url, {
            'prediction_batch_id': batch.id,
            'job_id': job.id
        })
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['prediction_status'], 'pending')
        self.assertEqual(response.data['job_status'], 'pending')
        self.assertEqual(response.data['completed'], False)
    
    def test_monitoring_dashboard_api(self):
        """Test the monitoring dashboard API endpoint."""
        url = reverse('monitoring_dashboard_api')
        response = self.client.get(url)
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn('counts', response.data)
        self.assertIn('recent_activity', response.data)
        self.assertEqual(response.data['counts']['students'], 1)
        self.assertEqual(response.data['counts']['courses'], 1)
        self.assertEqual(response.data['counts']['topics'], 2)
        self.assertEqual(response.data['counts']['learning_paths'], 1)
    
    def test_recommendation_dashboard_api(self):
        """Test the recommendation dashboard API endpoint."""
        url = reverse('recommendation_dashboard_api', args=[self.course.course_id])
        response = self.client.get(url)
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn('topic_stats', response.data)
        self.assertIn('path_stats', response.data)
        self.assertEqual(response.data['course']['id'], self.course.course_id)
    
    def test_api_documentation(self):
        """Test the API documentation endpoint."""
        url = reverse('api_documentation')
        response = self.client.get(url)
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn('endpoints', response.data)
        self.assertIn('api_version', response.data)
    
    def test_unauthenticated_access(self):
        """Test that unauthenticated access is blocked."""
        # Create a new client without authentication
        client = APIClient()
        
        url = reverse('student_profile_api', args=[self.student.student_id])
        response = client.get(url, {'course_id': self.course.course_id})
        
        self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED)


class APIIntegrationTests(TestCase):
    """Test class for API integration endpoints."""
    
    def setUp(self):
        """Set up test data."""
        # Create a user
        self.user = User.objects.create_user(
            username='testuser',
            password='testpassword'
        )
        
        # Create test data
        self.student = Student.objects.create(
            student_id='S12345',
            major='Computer Science',
            academic_level='freshman',
            gpa=3.5,
            study_frequency='daily',
            attendance_rate=90.0,
            participation_score=85.0
        )
        
        self.course = Course.objects.create(
            course_id='CS101',
            title='Introduction to Java Programming',
            description='A beginner course on Java programming'
        )
        
        # Set up API client
        self.client = APIClient()
        self.client.force_authenticate(user=self.user)
    
    def test_generate_and_check_recommendations(self):
        """Test the full flow of generating and checking recommendations."""
        # 1. Create a model and path generator
        model = KnowledgeTracingModel.objects.create(
            name='Test DKT Model',
            model_type='dkt',
            description='Test model',
            created_by=self.user,
            model_path='/test/path/model.pt',
            status='active',
            is_default=True,
            course=self.course
        )
        
        generator = PathGenerator.objects.create(
            name='Default Generator',
            description='Default path generator',
            created_by=self.user,
            is_active=True
        )
        
        # 2. Generate recommendations
        url = reverse('generate_recommendations_api', args=[self.student.student_id])
        response = self.client.post(url, {'course_id': self.course.course_id})
        
        self.assertEqual(response.status_code, status.HTTP_202_ACCEPTED)
        prediction_batch_id = response.data['prediction_batch_id']
        job_id = response.data['job_id']
        
        # 3. Update the status to simulate completion
        batch = PredictionBatch.objects.get(pk=prediction_batch_id)
        batch.status = 'completed'
        batch.completed_at = timezone.now()
        batch.save()
        
        job = PathGenerationJob.objects.get(pk=job_id)
        job.status = 'completed'
        job.completed_at = timezone.now()
        job.save()
        
        # Create a learning path for the job
        path = LearningPath.objects.create(
            generation_job=job,
            student=self.student,
            course=self.course,
            name='Test Learning Path',
            status='active',
            generated_at=timezone.now()
        )
        
        # 4. Check recommendation status
        url = reverse('check_recommendation_status_api')
        response = self.client.get(url, {
            'prediction_batch_id': prediction_batch_id,
            'job_id': job_id
        })
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['prediction_status'], 'completed')
        self.assertEqual(response.data['job_status'], 'completed')
        self.assertEqual(response.data['completed'], True)
        self.assertEqual(response.data['path_id'], path.id)