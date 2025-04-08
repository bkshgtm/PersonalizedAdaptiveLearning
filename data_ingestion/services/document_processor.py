import os
import logging
import tempfile
import datetime
import traceback # Import traceback for detailed error logging
from typing import List, Dict, Any, Tuple

from django.db import transaction, IntegrityError # Import IntegrityError
from django.utils import timezone

# For Dissect integration
from topic_classification.services.dissect_client import DissectClient
from topic_classification.services.classifier_factory import get_classifier
from core.services.answer_validation import AnswerValidator

from core.models import Student, Course, Assessment, Question, StudentInteraction, Topic
from data_ingestion.models import DocumentUpload, ExtractedQuestion, ProcessingLog

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Service to process document uploads containing questions and answers."""
    
    def __init__(self, document_id: int):
        """Initialize with a DocumentUpload ID."""
        self.document = DocumentUpload.objects.get(pk=document_id)
        self.dissect_client = DissectClient()
        self.answer_validator = AnswerValidator()
        self.errors = []
        
    def log_message(self, message: str, level: str = 'info') -> None:
        """Log a message to the processing log and Python logger."""
        ProcessingLog.objects.create(
            document_upload=self.document,
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
        """Update the status of the document upload."""
        self.document.status = status
        
        if error_message:
            self.document.error_message = error_message
            
        self.document.save()
    
    def process(self) -> bool:
        """Process the document file and extract questions/answers."""
        try:
            self.update_status('processing')
            self.log_message(f"Starting to process document: {self.document.file.name}")
            
            # Extract questions and answers based on file type
            if self.document.file.name.lower().endswith('.pdf'):
                questions = self._process_pdf()
            elif self.document.file.name.lower().endswith('.docx'):
                questions = self._process_docx()
            elif self.document.file.name.lower().endswith('.txt'):
                questions = self._process_text()
            else:
                raise ValueError(f"Unsupported file type: {self.document.file.name}")
            
            # Save extracted questions
            self._save_questions(questions)
            
            # Process each question (classify and validate)
            self._process_questions()
            
            # Update learning paths
            self._update_learning_paths()
            
            self.log_message(f"Processing completed. {self.document.questions_processed} questions processed, "
                            f"{self.document.questions_failed} questions failed.")
            
            if self.document.questions_failed > 0:
                status = 'completed with errors'
            else:
                status = 'completed'
                
            self.update_status(status)
            return True
            
        except Exception as e:
            error_message = f"Error processing document: {str(e)}"
            self.log_message(error_message, 'error')
            self.update_status('failed', error_message)
            return False
    
    def _process_pdf(self) -> List[Dict[str, Any]]:
        """Extract questions and answers from a PDF file using optimized text extraction."""
        import pdfplumber
        from concurrent.futures import ThreadPoolExecutor
        
        questions = []
        with self.document.file.open('rb') as file:
            with pdfplumber.open(file) as pdf:
                self.log_message(f"Processing PDF with {len(pdf.pages)} pages")
                
                # Process pages in parallel
                def process_page(page):
                    try:
                        text = page.extract_text()
                        if text:
                            return self._extract_qa_from_text(text, page_num=page.page_number)
                        return []
                    except Exception as e:
                        self.log_message(f"Error processing page {page.page_number}: {str(e)}", 'error')
                        return []
                
                with ThreadPoolExecutor(max_workers=4) as executor:
                    results = executor.map(process_page, pdf.pages)
                    for page_questions in results:
                        questions.extend(page_questions)
        
        return questions
    
    def _process_docx(self) -> List[Dict[str, Any]]:
        """Extract questions and answers from a DOCX file using memory-efficient streaming."""
        try:
            import sys
            self.log_message(f"Python path before docx import: {sys.path}", 'debug')
            import docx
            self.log_message(f"Successfully imported docx from: {docx.__file__}", 'debug')
            self.log_message(f"Python path after docx import: {sys.path}", 'debug')
        except ImportError as e:
            self.log_message(f"Failed to import docx: {str(e)}", 'error')
            self.log_message(f"Python path: {sys.path}", 'error')
            raise
        from io import BytesIO
        
        questions = []
        with self.document.file.open('rb') as file:
            # Use BytesIO to avoid temporary files
            doc = docx.Document(BytesIO(file.read()))
            
            # Process paragraphs in chunks
            text_chunks = []
            current_chunk = ""
            
            for para in doc.paragraphs:
                if para.text.strip():
                    current_chunk += para.text + "\n"
                    if len(current_chunk) > 10000:  # Process in 10KB chunks
                        text_chunks.append(current_chunk)
                        current_chunk = ""
            
            if current_chunk:
                text_chunks.append(current_chunk)
            
            # Process each chunk
            for chunk in text_chunks:
                questions.extend(self._extract_qa_from_text(chunk))
        
        return questions
    
    def _process_text(self) -> List[Dict[str, Any]]:
        """Extract questions and answers from a text file."""
        with self.document.file.open('r') as file:
            text = file.read()
            return self._extract_qa_from_text(text)
    
    def _extract_qa_from_text(self, text: str, page_num: int = 1) -> List[Dict[str, Any]]:
        """
        Extract questions and answers from text using DeepSeek.
        This is a more advanced approach using AI to identify Q&A pairs.
        """
        # Use DeepSeek to extract questions and answers
        prompt = f"""Analyze this student submission document to extract ALL question-answer pairs, regardless of formatting. The document may contain:
- Any combination of questions and answers
- Questions may be numbered, bulleted, or plain text
- Answers may appear immediately after questions, on separate lines, or with various prefixes
- Mixed formatting (bold, italics, code blocks, etc.)

CRITICAL REQUIREMENTS:
1. Extract EVERY identifiable question-answer pair, even if formatting is inconsistent
2. For each pair:
   QUESTION: [Full question text exactly as it appears]
   ANSWER: [Full student answer exactly as it appears]
3. Preserve ALL original formatting including:
   - Line breaks, indentation, spacing
   - Code blocks (with ``` markers)
   - Special characters, bullet points, numbering
4. Be extremely flexible with document structure but strict with output format
5. If ANY text resembles a question with an answer, extract it

EXAMPLES OF VALID OUTPUT FOR VARIOUS FORMATS:
1. Numbered question:
QUESTION: 1. Explain polymorphism in Java
ANSWER: Polymorphism allows objects to take many forms through method overriding.

2. Unformatted question:
QUESTION: What is encapsulation?
ANSWER: Hiding internal state and requiring interaction through methods.

3. Code answer:
QUESTION: Show a Java class example:
ANSWER: ```java
public class Example {{
    private String name;
}}
```

4. Bullet point answer:
QUESTION: List collection types:
ANSWER: - ArrayList
- HashMap
- HashSet

5. No clear prefix:
QUESTION: Difference between interface and abstract class?
ANSWER: Interfaces have no implementation while abstract classes can have both.

Now process this document, extracting ALL possible question-answer pairs:

--- START TEXT ---
{text}
--- END TEXT ---"""
        
        messages = [
            {"role": "system", "content": "You are an expert at identifying questions and answers in educational documents."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            # Log the exact prompt being sent
            self.log_message(f"Sending prompt to Dissect:\n---\n{prompt}\n---", 'debug')

            # Call Dissect API
            response = self.dissect_client._call_api(messages)
            self.log_message(f"Full Dissect API response:\n---\n{response}\n---", 'debug')

            if not response or 'choices' not in response or len(response['choices']) == 0:
                raise ValueError("Empty or invalid response from Dissect API")

            content = response['choices'][0]['message']['content']
            self.log_message(f"Raw AI response content:\n---\n{content}\n---", 'info')

            if not content.strip():
                self.log_message("Empty content received from Dissect", 'warning')
                return []

            # Parse the response to extract Q&A pairs
            questions = []
            lines = content.split('\n')
            i = 0
            question_number = 1
            while i < len(lines):
                line = lines[i].strip()
                if line.startswith('QUESTION:'):
                    # Start collecting question lines
                    current_question_lines = [line[len('QUESTION:'):].strip()]
                    i += 1
                    # Collect subsequent lines until ANSWER:, next QUESTION:, or end of content
                    while i < len(lines) and not lines[i].strip().startswith('ANSWER:') and not lines[i].strip().startswith('QUESTION:'):
                        current_question_lines.append(lines[i]) # Keep original line for formatting
                        i += 1

                    # Now look for the answer
                    current_answer_lines = []
                    if i < len(lines) and lines[i].strip().startswith('ANSWER:'):
                        current_answer_lines.append(lines[i].strip()[len('ANSWER:'):].strip())
                        i += 1
                        # Collect subsequent lines until next QUESTION: or end of content
                        while i < len(lines) and not lines[i].strip().startswith('QUESTION:'):
                            current_answer_lines.append(lines[i]) # Keep original line for formatting
                            i += 1

                    # If we found both parts, add the question
                    if current_question_lines and current_answer_lines:
                        questions.append({
                            'question_text': "\n".join(current_question_lines).strip(),
                            'student_answer': "\n".join(current_answer_lines).strip(),
                            'question_number': question_number,
                            'page_number': page_num
                        })
                        question_number += 1
                    # If loop ended without finding ANSWER:, or if QUESTION wasn't followed by lines,
                    # we just skip this potential block and the outer loop continues from index i.
                    # This handles cases where parsing might fail for a block.
                else:
                    # Skip lines that are not the start of a QUESTION block
                    i += 1

            self.log_message(f"Extracted {len(questions)} questions from page {page_num}")
            return questions

        except Exception as e:
            self.log_message(f"Error extracting Q&A with Dissect: {str(e)}", 'error')
            # Fallback to simple extraction if Dissect fails
            return self._simple_qa_extraction(text, page_num)
    
    def _simple_qa_extraction(self, text: str, page_num: int = 1) -> List[Dict[str, Any]]:
        """
        Simple rule-based extraction of questions and answers.
        This is a fallback method if AI extraction fails.
        """
        questions = []
        lines = text.split('\n')
        
        question_number = 1
        current_question = None
        current_answer = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Look for question indicators
            if (line.startswith(f"{question_number}.") or 
                line.startswith(f"Q{question_number}.") or 
                line.startswith(f"Question {question_number}:")):
                
                # Save previous question if exists
                if current_question and current_answer:
                    questions.append({
                        'question_text': current_question,
                        'student_answer': current_answer,
                        'question_number': question_number - 1,
                        'page_number': page_num
                    })
                
                current_question = line
                current_answer = ""
                question_number += 1
            elif line.startswith("A:") or line.startswith("Answer:"):
                # This is an answer
                current_answer = line.split(":", 1)[1].strip()
            elif current_question and not current_answer:
                # Continue building the question
                current_question += " " + line
            elif current_question and current_answer:
                # Continue building the answer
                current_answer += " " + line
        
        # Don't forget the last question
        if current_question and current_answer:
            questions.append({
                'question_text': current_question,
                'student_answer': current_answer,
                'question_number': question_number - 1,
                'page_number': page_num
            })
        
        return questions
    
    def _save_questions(self, questions: List[Dict[str, Any]]) -> None:
        """Save extracted questions to the database."""
        for q_data in questions:
            ExtractedQuestion.objects.create(
                document=self.document,
                question_text=q_data['question_text'],
                student_answer=q_data['student_answer'],
                question_number=q_data['question_number'],
                page_number=q_data['page_number']
            )
    
    def _process_questions(self) -> None:
        """Process each extracted question (classify and validate)."""
        # Try to get any active classifier
        from topic_classification.models import ClassificationModel
        classifier = None
        
        # First try to get assessment-specific classifier if available
        if hasattr(self.document.assessment, 'classification_model'):
            classifier = get_classifier(self.document.assessment.classification_model.id)
        
        # Fall back to default classifier
        if not classifier:
            default_model = ClassificationModel.objects.filter(is_default=True).first()
            if default_model:
                classifier = get_classifier(default_model.id)
        
        if not classifier:
            self.log_message("No active classification model found - skipping classification", 'warning')
            return
        
        # Get all questions for this document
        questions = ExtractedQuestion.objects.filter(document=self.document, is_processed=False)
        
        for extracted_q in questions:
            question = None # Initialize question to None
            try:
                # 1. Create or get a Question object
                question, created = Question.objects.get_or_create(
                    text=extracted_q.question_text,
                    defaults={
                        'question_id': f"doc-{self.document.id}-q-{extracted_q.id}",
                        'assessment': self.document.assessment,
                        'question_type': 'short_answer'  # Using a valid choice from QUESTION_TYPE_CHOICES
                    }
                )
                
                # 2. Classify the question to get a topic
                topic, confidence = classifier.classify_question(question)
                
                # 3. Validate the student's answer with added logging and error handling
                validation_result = None
                try:
                    self.log_message(f"Calling AnswerValidator for QID {extracted_q.id} (Question PK: {question.pk if question else 'N/A'})", 'debug')
                    self.log_message(f"  - Question Text: {question.text[:100]}...", 'debug')
                    self.log_message(f"  - Student Answer: {extracted_q.student_answer[:100]}...", 'debug')
                    
                    validation_result = self.answer_validator.validate_answer(
                        question=question,
                        student_answer=extracted_q.student_answer
                    )
                    self.log_message(f"Received validation result for QID {extracted_q.id}: {validation_result}", 'debug')

                except Exception as validation_error:
                    error_trace = traceback.format_exc()
                    self.log_message(f"Exception during AnswerValidator call for QID {extracted_q.id}: {validation_error}\nTrace: {error_trace}", 'error')
                    # Create a default error structure if validation failed completely
                    validation_result = {
                        'is_correct': False,
                        'score': 0.0,
                        'feedback': f"Validation Error: {validation_error}",
                        '_detailed': self.answer_validator._create_default_error_response(f"Validation exception: {validation_error}")
                    }

                # Ensure validation_result is not None before proceeding
                if validation_result is None:
                     self.log_message(f"Validation result is None for QID {extracted_q.id} after try/except block.", 'error')
                     # Use a default error structure
                     validation_result = {
                        'is_correct': False,
                        'score': 0.0,
                        'feedback': "Validation Error: Unknown validation failure",
                        '_detailed': self.answer_validator._create_default_error_response("Unknown validation failure")
                     }

                # Store both basic and detailed validation results
                extracted_q.topic = topic
                extracted_q.is_correct = validation_result['is_correct']
                extracted_q.confidence_score = validation_result['score']
                extracted_q.feedback = validation_result['feedback']
                
                # Store detailed validation if available
                if '_detailed' in validation_result:
                    detailed = validation_result['_detailed']
                    extracted_q.validation_metadata = {
                        'scores': detailed['score'],
                        'feedback': detailed['feedback'],
                        'concepts': detailed['concepts']
                    }
                    self.log_message(f"QID {extracted_q.id}: Detailed validation - Accuracy={detailed['score']['accuracy']:.2f}, Completeness={detailed['score']['completeness']:.2f}", 'debug')

                self.log_message(f"QID {extracted_q.id}: Validation - Correct={validation_result['is_correct']}, Score={validation_result['score']:.2f}", 'debug')
                extracted_q.is_processed = True
                extracted_q.save()
                
                # 5. Create or update a student interaction record safely
                if self.document.student and self.document.course and question:
                    try:
                        # Calculate the next attempt number for this student and question
                        last_attempt = StudentInteraction.objects.filter(
                            student=self.document.student,
                            question=question
                        ).order_by('-attempt_number').first()
                        
                        next_attempt_number = (last_attempt.attempt_number + 1) if last_attempt else 1
                        
                        # Use student_id directly for logging and query
                        student_id_for_log = self.document.student_id if self.document.student_id else "N/A"
                        self.log_message(f"Attempting to create/update interaction for QID {extracted_q.id}, Student ID {student_id_for_log}, Attempt {next_attempt_number}", 'debug')

                        # Use update_or_create with student_id
                        interaction, created = StudentInteraction.objects.update_or_create(
                            student_id=self.document.student_id, # Use student_id
                            question=question,
                            attempt_number=next_attempt_number, 
                            defaults={
                                'response': extracted_q.student_answer, # Always update response
                                'correct': validation_result['is_correct'], # Update correctness
                                'score': validation_result['score'], # Update score
                                'time_taken': datetime.timedelta(minutes=5),  # Default or update if needed
                                'timestamp': timezone.now(), # Update timestamp
                                'resource_viewed_before': False # Default or update if needed
                            }
                        )
                        log_action = "Created" if created else "Updated"
                        # Use student_id directly for logging
                        self.log_message(f"{log_action} interaction record for QID {extracted_q.id}, Student ID {student_id_for_log}, Attempt {next_attempt_number}", 'info')

                    except Exception as interaction_error:
                         # Catch any error during interaction creation/update
                         error_trace = traceback.format_exc()
                         self.log_message(f"Error creating/updating interaction for QID {extracted_q.id}: {interaction_error}\nTrace: {error_trace}", 'error')

                
                self.document.questions_processed += 1
                self.document.save()
                
            except Exception as e:
                error_trace = traceback.format_exc()
                self.log_message(f"Error processing question {extracted_q.id} (Question PK: {question.pk if question else 'N/A'}): {str(e)}\nTrace: {error_trace}", 'error')
                # Ensure extracted_q is marked as processed even if error occurs after question creation
                if extracted_q and not extracted_q.is_processed:
                     extracted_q.is_processed = True # Mark as processed to avoid retrying indefinitely
                     extracted_q.feedback = f"Processing Error: {str(e)}" # Add error feedback
                     extracted_q.save()
                self.document.questions_failed += 1
                self.document.save() # Save failure count immediately
    
    def _update_learning_paths(self) -> None:
        """Update learning paths based on processed questions."""
        if not self.document.student or not self.document.course:
            self.log_message("Cannot update learning paths: student or course not specified", 'warning')
            return
        
        try:
            # Import here to avoid circular imports
            from learning_paths.tasks import generate_learning_path
            from ml_models.tasks import generate_mastery_predictions
            from ml_models.models import KnowledgeTracingModel, PredictionBatch
            from learning_paths.models import PathGenerationJob, PathGenerator
            from knowledge_graph.models import KnowledgeGraph
            
            # Get the default model for the course
            model = KnowledgeTracingModel.objects.filter(
                course=self.document.course, 
                is_default=True, 
                status='active'
            ).first()
            
            if not model:
                self.log_message("No active knowledge tracing model found for this course", 'warning')
                return
            
            # Create a prediction batch
            prediction_batch = PredictionBatch.objects.create(
                model=model,
                status='pending'
            )
            
            # Start the prediction task
            generate_mastery_predictions.delay(prediction_batch.id)
            
            # Get the active path generator
            generator = PathGenerator.objects.filter(is_active=True).first()
            
            if not generator:
                self.log_message("No active path generator found", 'warning')
                return
            
            # Get the active knowledge graph
            knowledge_graph = KnowledgeGraph.objects.filter(is_active=True).first()
            
            # Create a path generation job
            job = PathGenerationJob.objects.create(
                generator=generator,
                student=self.document.student,
                course=self.document.course,
                status='pending',
                prediction_batch=prediction_batch,
                knowledge_graph=knowledge_graph
            )
            
            # Start the generation task
            generate_learning_path.delay(job.id)
            
            self.log_message(f"Learning path update initiated: job ID {job.id}")
            
        except Exception as e:
            self.log_message(f"Error updating learning path: {str(e)}", 'error')
