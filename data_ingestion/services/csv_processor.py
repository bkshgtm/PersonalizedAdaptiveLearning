import csv
import io
import logging
import datetime
from typing import Dict, List, Tuple, Any

from django.db import transaction
from django.utils import timezone

from core.models import (
    Student, Course, Assessment, Question, StudentInteraction
)
from data_ingestion.models import DataUpload, ProcessingLog

logger = logging.getLogger(__name__)

class CSVProcessor:
    """
    Service class to process CSV uploads containing student learning data.
    """
    
    def __init__(self, data_upload_id: int):
        """
        Initialize with the ID of a DataUpload record.
        """
        self.data_upload = DataUpload.objects.get(pk=data_upload_id)
        self.errors = []
        self.records_processed = 0
        self.records_failed = 0
        
    def log_message(self, message: str, level: str = 'info') -> None:
        """
        Log a message to the processing log and Python logger.
        """
        ProcessingLog.objects.create(
            data_upload=self.data_upload,
            message=message,
            level=level
        )
        
        if level == 'info':
            logger.info(message)
        elif level == 'warning':
            logger.warning(message)
        elif level == 'error':
            logger.error(message)
    
    def update_status(self, status: str, error_message: str = '') -> None:
        """
        Update the status of the data upload.
        """
        self.data_upload.status = status
        self.data_upload.rows_processed = self.records_processed
        self.data_upload.rows_failed = self.records_failed
        
        if error_message:
            self.data_upload.error_message = error_message
            
        self.data_upload.save()
    
    def process(self) -> bool:
        """
        Process the CSV file and create records in the database.
        Returns True if processing was successful, False otherwise.
        """
        try:
            self.update_status('processing')
            self.log_message(f"Starting to process file: {self.data_upload.file.name}")
            
            # Read the CSV file
            csv_file = self.data_upload.file.open('r')
            csv_reader = csv.DictReader(io.StringIO(csv_file.read().decode('utf-8')))
            
            # Validate the CSV structure
            required_fields = [
                'student_id', 'course_id', 'question_id', 'correct',
                'assessment_id', 'assessment_type'
            ]
            
            # Check if required fields are present
            field_names = csv_reader.fieldnames
            if not field_names:
                raise ValueError("CSV file has no headers")
                
            missing_fields = [field for field in required_fields if field not in field_names]
            if missing_fields:
                error_msg = f"CSV file is missing required fields: {', '.join(missing_fields)}"
                self.log_message(error_msg, 'error')
                self.update_status('failed', error_msg)
                return False
            
            # Reset the file pointer to the beginning
            csv_file.seek(0)
            csv_reader = csv.DictReader(io.StringIO(csv_file.read().decode('utf-8')))
            
            # Process each row in the CSV file
            with transaction.atomic():
                for row_num, row in enumerate(csv_reader, start=1):
                    try:
                        self._process_row(row)
                        self.records_processed += 1
                        
                        # Log progress every 100 rows
                        if row_num % 100 == 0:
                            self.log_message(f"Processed {row_num} rows")
                            self.update_status('processing')
                    
                    except Exception as e:
                        self.log_message(f"Error processing row {row_num}: {str(e)}", 'error')
                        self.records_failed += 1
            
            self.log_message(
                f"Processing completed. {self.records_processed} records processed, "
                f"{self.records_failed} records failed."
            )
            
            if self.records_failed > 0:
                status = 'completed with errors'
            else:
                status = 'completed'
                
            self.update_status(status)
            return True
            
        except Exception as e:
            error_message = f"Error processing CSV file: {str(e)}"
            self.log_message(error_message, 'error')
            self.update_status('failed', error_message)
            return False
        finally:
            if csv_file and not csv_file.closed:
                csv_file.close()
    
    def _process_row(self, row: Dict[str, str]) -> None:
        """
        Process a single row from the CSV file.
        """
        # Extract student data
        student = self._get_or_create_student(row)
        
        # Extract course data
        course = self._get_or_create_course(row)
        
        # Add student to course if not already added
        if student not in course.students.all():
            course.students.add(student)
        
        # Extract assessment data
        assessment = self._get_or_create_assessment(row, course)
        
        # Extract question data
        question = self._get_or_create_question(row, assessment)
        
        # Create student interaction
        self._create_student_interaction(row, student, question)
    
    def _get_or_create_student(self, row: Dict[str, str]) -> Student:
        """
        Get or create a Student record based on the CSV row.
        """
        student_id = row['student_id']
        
        # Try to get the student
        try:
            return Student.objects.get(student_id=student_id)
        except Student.DoesNotExist:
            # Create a new student
            student_data = {
                'student_id': student_id,
                'major': row.get('major', 'Unknown'),
                'academic_level': row.get('academic_level', 'freshman'),
                'gpa': float(row.get('GPA', 0.0)),
                'prior_knowledge_score': float(row.get('prior_knowledge_score', 0.0)) if row.get('prior_knowledge_score') else None,
                'study_frequency': row.get('study_frequency', 'weekly'),
                'attendance_rate': float(row.get('attendance_rate', 0.0)),
                'participation_score': float(row.get('participation_score', 0.0)),
                'last_login_date': self._parse_datetime(row.get('last_login_date')) if row.get('last_login_date') else None,
                'total_time_spent': datetime.timedelta(minutes=float(row.get('total_time_spent_on_platform', 0))) if row.get('total_time_spent_on_platform') else None,
                'average_time_per_session': datetime.timedelta(minutes=float(row.get('average_time_per_session', 0))) if row.get('average_time_per_session') else None,
            }
            
            student = Student.objects.create(**student_data)
            self.log_message(f"Created new student with ID: {student_id}")
            return student
    
    def _get_or_create_course(self, row: Dict[str, str]) -> Course:
        """
        Get or create a Course record based on the CSV row.
        """
        course_id = row['course_id']
        
        # Try to get the course
        try:
            return Course.objects.get(course_id=course_id)
        except Course.DoesNotExist:
            # Create a new course
            course_data = {
                'course_id': course_id,
                'title': row.get('course_title', f"Course {course_id}"),
                'description': row.get('course_description', ''),
            }
            
            course = Course.objects.create(**course_data)
            self.log_message(f"Created new course with ID: {course_id}")
            return course
    
    def _get_or_create_assessment(self, row: Dict[str, str], course: Course) -> Assessment:
        """
        Get or create an Assessment record based on the CSV row.
        """
        assessment_id = row['assessment_id']
        
        # Try to get the assessment
        try:
            return Assessment.objects.get(assessment_id=assessment_id)
        except Assessment.DoesNotExist:
            # Create a new assessment
            assessment_data = {
                'assessment_id': assessment_id,
                'title': row.get('assessment_title', f"Assessment {assessment_id}"),
                'assessment_type': row.get('assessment_type', 'quiz'),
                'course': course,
                'date': self._parse_datetime(row.get('assessment_date')) if row.get('assessment_date') else timezone.now(),
                'proctored': row.get('proctored', 'false').lower() == 'true',
            }
            
            assessment = Assessment.objects.create(**assessment_data)
            self.log_message(f"Created new assessment with ID: {assessment_id}")
            return assessment
    
    def _get_or_create_question(self, row: Dict[str, str], assessment: Assessment) -> Question:
        """
        Get or create a Question record based on the CSV row.
        """
        question_id = row['question_id']
        
        # Try to get the question
        try:
            return Question.objects.get(question_id=question_id)
        except Question.DoesNotExist:
            # Create a new question
            question_data = {
                'question_id': question_id,
                'assessment': assessment,
                'text': row.get('question_text', ''),
                'question_type': row.get('question_type', 'mcq'),
                # Topic will be set later by the topic classification service
                'topic': None,
            }
            
            question = Question.objects.create(**question_data)
            self.log_message(f"Created new question with ID: {question_id}")
            return question
    
    def _create_student_interaction(self, row: Dict[str, str], student: Student, question: Question) -> StudentInteraction:
        """
        Create a StudentInteraction record based on the CSV row.
        """
        # Parse boolean values
        correct = row.get('correct', '0')
        if isinstance(correct, str):
            correct = correct.lower() in ('1', 'true', 'yes', 't', 'y')
        else:
            correct = bool(correct)
            
        resource_viewed = row.get('resource_viewed_before_question', 'false')
        if isinstance(resource_viewed, str):
            resource_viewed = resource_viewed.lower() in ('true', 'yes', 't', 'y', '1')
        else:
            resource_viewed = bool(resource_viewed)
        
        # Create student interaction
        interaction_data = {
            'student': student,
            'question': question,
            'response': row.get('student_response', ''),
            'correct': correct,
            'score': float(row.get('score', 0.0)) if row.get('score') else None,
            'time_taken': datetime.timedelta(seconds=float(row.get('time_taken', 0))),
            'timestamp': self._parse_datetime(row.get('timestamp')) if row.get('timestamp') else timezone.now(),
            'attempt_number': int(row.get('attempt_number', 1)),
            'resource_viewed_before': resource_viewed,
        }
        
        interaction = StudentInteraction.objects.create(**interaction_data)
        return interaction
    
    def _parse_datetime(self, date_str: str) -> datetime.datetime:
        """
        Parse a datetime string in various formats.
        """
        if not date_str:
            return timezone.now()
            
        formats = [
            '%Y-%m-%d %H:%M:%S',
            '%Y-%m-%d %H:%M',
            '%Y-%m-%d',
            '%m/%d/%Y %H:%M:%S',
            '%m/%d/%Y %H:%M',
            '%m/%d/%Y',
            '%d/%m/%Y %H:%M:%S',
            '%d/%m/%Y %H:%M',
            '%d/%m/%Y',
        ]
        
        for fmt in formats:
            try:
                dt = datetime.datetime.strptime(date_str, fmt)
                # Make timezone-aware if naive
                if dt.tzinfo is None:
                    dt = timezone.make_aware(dt)
                return dt
            except ValueError:
                continue
                
        # If all parsing attempts fail, return current time
        self.log_message(f"Could not parse datetime: {date_str}", 'warning')
        return timezone.now()