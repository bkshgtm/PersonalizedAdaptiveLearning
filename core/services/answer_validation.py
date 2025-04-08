# core/services/answer_validation.py

import json
import logging
from typing import Dict, Any

from core.models import Question, StudentInteraction
from topic_classification.services.dissect_client import DissectClient

logger = logging.getLogger(__name__)

class AnswerValidator:
    """
    Service for validating student answers.
    """
    
    def __init__(self):
        """
        Initialize the answer validator.
        """
        self.client = DissectClient()

    def validate_answer(self, question: Question, student_answer: str) -> Dict[str, Any]:
        """
        Validate a student's answer with detailed scoring and feedback.
        
        Args:
            question: The question being answered (must have an assessment)
            student_answer: The student's submitted answer
            
        Returns:
            Dictionary with validation results including:
            - Detailed scoring breakdown
            - Structured feedback
            - Backwards-compatible fields
            
        Raises:
            ValueError: If question has no associated assessment
        """
        if not question.assessment:
            raise ValueError("Question must have an associated assessment")
        try:
            # Get detailed validation from Dissect
            detailed = self._validate_detailed(question, student_answer)

            # Use the new top-level feedback summary if available, otherwise fallback
            default_suggestions = ['No feedback available']
            feedback_details_dict = detailed.get('feedback') 
            if isinstance(feedback_details_dict, dict):
                default_suggestions = feedback_details_dict.get('suggestions', default_suggestions)
            
            feedback_summary = detailed.get('feedback', "\n".join(default_suggestions)) 

            # Ensure consistent response structure
            if 'validation_details' not in detailed:
                detailed['validation_details'] = {
                    'concepts': detailed.get('concepts', {'covered': [], 'missing': []}),
                    'scores': detailed.get('scores', {
                        'accuracy': detailed.get('score', 0.0),
                        'completeness': 1.0 if detailed.get('is_correct', False) else 0.0,
                        'clarity': 1.0,
                        'overall': detailed.get('score', 0.0)
                    })
                }
            elif 'concepts' not in detailed['validation_details']:
                detailed['validation_details']['concepts'] = {'covered': [], 'missing': []}
            
            return {
                'is_correct': detailed.get('is_correct', detailed['score']['overall'] > 0.7),
                'score': detailed['score']['overall'],
                'feedback': feedback_summary,
                '_detailed': detailed
            }
        except Exception as e:
            logger.error(f"Error validating answer: {str(e)}")
            return {
                'is_correct': False,
                'score': 0.0,
                'feedback': "Validation error occurred",
                '_detailed': {
                    'score': {'overall': 0.0},
                    'feedback': {
                        'suggestions': ['Validation system error']
                    }
                }
            }

    def _validate_detailed(self, question: Question, student_answer: str) -> Dict[str, Any]:
        """
        Perform detailed answer validation with scoring breakdown.
        Uses the Dissect client's check_answer_correctness method.
        """
        try:
            result = self.client.check_answer_correctness(
                question_text=question.text,
                correct_answer=None,
                student_answer=student_answer
            )
        except Exception as e:
            logger.error(f"Error calling Dissect client: {str(e)}")
            return self._create_default_error_response(f"Dissect client error: {e}")
            
        logger.debug(f"Raw Dissect validation result received by AnswerValidator: {result}")

        parsed_result = None
        if isinstance(result, str):
            try:
                parsed_result = json.loads(result)
                logger.debug("Successfully parsed string response as JSON.")
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse string response as JSON: {e}")
                return self._create_default_error_response(f"Response parsing failed (string): {e}") 
        elif isinstance(result, dict):
            parsed_result = result
            logger.debug("Result is already a dictionary.")
        else:
            logger.error(f"Unexpected result type from DissectClient: {type(result)}")
            return self._create_default_error_response("Unexpected response type")

        if not parsed_result:
            logger.error("Parsed result is None after attempting parsing.")
            return self._create_default_error_response("Response parsing failed (None)")

        logger.debug(f"Parsed result before format check: {parsed_result}")

        if 'validation_details' in parsed_result and 'feedback' in parsed_result and 'explanation' in parsed_result:
            logger.debug("Parsing using new detailed format.")
            validation_details = parsed_result['validation_details']
            scores = validation_details.get('scores', {})
            
            # Get scores from Dissect response
            accuracy = min(1.0, max(0.0, float(scores.get('accuracy', 0.0))))
            completeness = min(1.0, max(0.0, float(scores.get('completeness', 0.0))))
            clarity = min(1.0, max(0.0, float(scores.get('clarity', 0.0))))
            overall = min(1.0, max(0.0, float(scores.get('overall', 0.0))))

            # Determine correctness based on overall score threshold
            is_correct = parsed_result.get('is_correct', overall >= 0.8)

            return {
                'is_correct': is_correct,
                'feedback': parsed_result.get('feedback', 'No feedback provided'),
                'explanation': parsed_result.get('explanation', 'No explanation provided'),
                'score': {
                    'accuracy': round(accuracy, 2),
                    'completeness': round(completeness, 2),
                    'clarity': round(clarity, 2),
                    'overall': round(overall, 2)
                }
            }
        else:
            logger.debug("Attempting to parse using fallback/old format.")
            # Expect Dissect to provide all scoring dimensions
            scores = parsed_result.get('scores', {})
            return {
                'is_correct': parsed_result.get('is_correct', False),
                'feedback': parsed_result.get('feedback', 'No feedback provided'),
                'explanation': parsed_result.get('explanation', 'No explanation provided'),
                'score': {
                    'accuracy': round(float(scores.get('accuracy', 0.0)), 2),
                    'completeness': round(float(scores.get('completeness', 0.0)), 2),
                    'clarity': round(float(scores.get('clarity', 0.0)), 2),
                    'overall': round(float(scores.get('overall', 0.0)), 2)
                }
            }

    def _create_default_error_response(self, error_message: str = "Validation system error") -> Dict[str, Any]:
        """Create a standardized error response with a specific message."""
        logger.warning(f"Creating default error response: {error_message}")
        weakness = f"Validation Error: {error_message}" 
        suggestion = "Please review the input or contact support."

        return {
            'score': {
                'accuracy': 0.0,
                'completeness': 0.0,
                'clarity': 0.0,
                'overall': 0.0
            },
            'feedback': {
                'strengths': [],
                'weaknesses': [weakness],
                'suggestions': [suggestion]
            },
            'concepts': {
                'covered': [],
                'missing': []
            }
        }
