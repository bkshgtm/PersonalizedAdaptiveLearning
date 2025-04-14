from celery import shared_task
import logging
from django.db import IntegrityError, transaction
from sympy import Max

from .services.csv_processor import CSVProcessor
from .services.document_processor import DocumentProcessor
from .models import DataUpload, DocumentUpload
from core.models import StudentInteraction

logger = logging.getLogger(__name__)

@shared_task
def process_csv_upload(data_upload_id):
    """
    Celery task to process a CSV upload.
    
    Args:
        data_upload_id: ID of the DataUpload instance to process
    """
    try:
        logger.info(f"Starting CSV processing task for upload {data_upload_id}")
        
        processor = CSVProcessor(data_upload_id)
        success = processor.process()
        
        if success:
            logger.info(f"CSV processing completed for upload {data_upload_id}")
        else:
            logger.error(f"CSV processing failed for upload {data_upload_id}")
        
        return success
    
    except Exception as e:
        logger.exception(f"Error in CSV processing task for upload {data_upload_id}: {str(e)}")
        
        # Update the data upload status to failed
        try:
            data_upload = DataUpload.objects.get(pk=data_upload_id)
            data_upload.status = 'failed'
            data_upload.error_message = str(e)
            data_upload.save()
        except Exception as update_error:
            logger.error(f"Failed to update DataUpload status: {str(update_error)}")
        
        # Re-raise the exception to mark the task as failed
        raise


@shared_task
def process_document_upload(document_id):
    """
    Celery task to process a document upload containing questions and answers.
    
    Args:
        document_id: ID of the DocumentUpload instance to process
    """
    try:
        document = DocumentUpload.objects.get(pk=document_id)
        logger.info(f"Processing document {document.file.name} (ID: {document_id})")
        
        processor = DocumentProcessor(document_id)
        success = processor.process()
        
        if success:
            logger.info(f"Successfully processed document {document.file.name} - extracted {document.questions_processed} questions")
        else:
            logger.error(f"Document processing failed for {document.file.name} - {document.error_message}")
        
        return success
    
    except Exception as e:
        document = DocumentUpload.objects.get(pk=document_id)
        logger.exception(f"Error processing document {document.file.name}: {str(e)}")
        
        # Update the document upload status to failed
        try:
            document = DocumentUpload.objects.get(pk=document_id)
            document.status = 'failed'
            document.error_message = str(e)
            document.save()
        except Exception as update_error:
            logger.error(f"Failed to update DocumentUpload status: {str(update_error)}")
        
        # Re-raise the exception to mark the task as failed
        raise

def create_student_interaction(student, question, answer, attempt_number):
    """
    Helper function to safely create student interaction records.
    Handles duplicate attempts by incrementing attempt_number.
    """
    with transaction.atomic():
        try:
            # Try to create new interaction
            interaction = StudentInteraction.objects.create(
                student=student,
                question=question,
                attempt_number=attempt_number,
                answer=answer
            )
            return interaction, True
        except IntegrityError:
            # If duplicate exists, find the next available attempt number
            max_attempt = StudentInteraction.objects.filter(
                student=student,
                question=question
            ).aggregate(Max('attempt_number'))['attempt_number__max'] or 0
            
            interaction = StudentInteraction.objects.create(
                student=student,
                question=question,
                attempt_number=max_attempt + 1,
                answer=answer
            )
            return interaction, False
