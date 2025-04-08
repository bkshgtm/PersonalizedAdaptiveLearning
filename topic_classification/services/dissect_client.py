# topic_classification/services/dissect_client.py

import requests
import json
import logging
import os
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)

class DissectClient:
    """
    Client for interacting with the Dissect API for AI capabilities.
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Dissect client.

        Args:
            api_key: Dissect API key. If None, will try to get from environment.
        """
        self.api_key = api_key or os.environ.get('DISSECT_API_KEY')
        if not self.api_key:
            raise ValueError("Dissect API key is required")

        self.api_base_url = os.environ.get('DISSECT_API_URL', 'https://api.dissect.com/v1') # Assuming similar URL structure
        self.model = os.environ.get('DISSECT_MODEL', 'dissect-coder-1.3b-instruct') # Assuming similar model naming

    def _call_api(self, messages: List[Dict[str, str]], temperature: float = 0.1) -> Dict[str, Any]:
        """
        Call the Dissect API.

        Args:
            messages: List of message objects with role and content
            temperature: Temperature parameter for generating responses
            
        Returns:
            API response as a dictionary
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        data = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": 1024
        }
        
        try:
            response = requests.post(
                f"{self.api_base_url}/chat/completions",
                headers=headers,
                data=json.dumps(data)
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error calling Dissect API: {str(e)}")
            if hasattr(e, 'response') and e.response:
                logger.error(f"Response status: {e.response.status_code}")
                logger.error(f"Response body: {e.response.text}")
            raise
    
    def classify_topic(self, question_text: str, available_topics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Classify a question into a topic using Dissect.

        Args:
            question_text: The question text to classify
            available_topics: List of available topics with their IDs and names
            
        Returns:
            Dictionary with topic ID, confidence, and explanation
        """
        # Format topics for the prompt
        topic_text = "\n".join([
            f"ID: {topic['id']}, Name: {topic['name']}, Description: {topic.get('description', '')}"
            for topic in available_topics
        ])
        
        messages = [
            {"role": "system", "content": "You are an expert in Java programming and educational content classification."},
            {"role": "user", "content": f"""
Classify the following Java programming question into the most appropriate topic.

Available topics:
{topic_text}

Question to classify:
{question_text}

Respond with a JSON object with the following structure:
{{
    "topic_id": "The ID of the most relevant topic",
    "confidence": "A number between 0 and 1 indicating your confidence",
    "explanation": "Brief explanation of why this topic is the best match"
}}
"""
            }
        ]
        
        response = self._call_api(messages)
        
        try:
            response_content = response['choices'][0]['message']['content']
            logger.debug(f"Raw Dissect API response content for check_answer_correctness:\n---\n{response_content}\n---") # Added logging
            # Extract JSON from the response
            import re
            json_match = re.search(r'\{.*\}', response_content, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group(0))
                return result
            else:
                # If no JSON found, try to parse the whole content
                return json.loads(response_content)
        except (KeyError, json.JSONDecodeError) as e:
            logger.error(f"Error parsing Dissect response: {str(e)}")
            logger.error(f"Response content: {response}")
            raise ValueError(f"Invalid response format from Dissect: {e}")

    def check_answer_correctness(self, question_text: str, correct_answer: str,
                              student_answer: str) -> Dict[str, Any]:
        """
        Check the correctness of a student's answer using Dissect.

        Args:
            question_text: The question text
            correct_answer: (Optional) The correct answer for reference. If None, AI must determine it.
            student_answer: The student's answer to evaluate
            
        Returns:
            Dictionary with correctness assessment, score, and feedback
        """
        # Determine the system prompt based on whether a correct answer is provided
        if correct_answer:
            system_content = """You are an expert in evaluating Java programming answers based on a provided correct answer. Your primary goal is to determine if the student understands the concept tested by the question by comparing their answer to the reference.

Evaluation Guidelines:
1.  **Code Output Questions:** If the question asks for the output of a code snippet, compare the student's answer ONLY to the expected printed output in the 'Correct Answer'. The student's answer is correct if it exactly matches the primary output. Ignore extra explanatory text in the 'Correct Answer' unless the question asks for it.
2.  **Conceptual Questions:** Allow for paraphrasing and variations in wording if the core concept matches the 'Correct Answer'.
3.  **Code Writing Questions:** Check for functional equivalence against the 'Correct Answer'.
4.  **Formatting:** Be lenient with minor differences in whitespace or formatting.
5.  **Focus:** Base your evaluation strictly on comparing the Student's Answer to the Correct Answer in the context of the Question.
"""
            user_content = f"""
Evaluate whether the student's answer to the following Java programming question is correct by comparing it to the provided correct answer.

Question:
```java
{question_text}
```

Correct Answer:
```java
{correct_answer}
```

Student's Answer:
```java
{student_answer}
```

Respond with a JSON object with the following structure:
{{
    "is_correct": true/false, # Whether the answer is fundamentally correct compared to the reference
    "score": a number between 0 and 1 indicating the quality/similarity to the reference,
    "feedback": "Detailed feedback on the student's answer comparing it to the correct one",
    "explanation": "Explanation of the score and assessment based on the comparison"
}}
"""
        else:
            # AI needs to determine the correct answer first
            system_content = """You are an expert Java programming teaching assistant. Your goal is to evaluate student answers like a helpful human teacher.

Your Task:
1.  **Determine Correctness:** First, determine the *correct and complete* answer for the given Java question based *only* on the question itself.
2.  **Evaluate Student Answer:** Compare the student's answer to the correct answer you determined. Evaluate based on conceptual understanding, accuracy, and completeness.
3.  **Provide Nuanced Feedback:** Generate detailed, constructive feedback.

Evaluation Guidelines:
*   **Correct Answers:** If the student's answer is correct or mostly correct:
    *   Set `is_correct` to `true`.
    *   Assign a score reflecting quality (e.g., 1.0 for perfect, 0.8-0.9 for minor omissions/clarity issues).
    *   **Identify Strengths:** List specific positive aspects (e.g., "Correctly identified the core concept," "Good use of terminology," "Code is efficient").
    *   Provide minor suggestions if applicable (e.g., "Could be slightly more detailed here," "Consider edge cases like...").
*   **Incorrect Answers:** If the student's answer is incorrect:
    *   Set `is_correct` to `false`.
    *   Assign a score reflecting the degree of error (e.g., 0.0-0.5).
    *   **Identify Weaknesses:** List specific errors or misunderstandings (e.g., "Incorrectly defined inheritance," "Missed the purpose of the 'final' keyword," "Code has a syntax error on line X").
    *   **Provide Suggestions:** Offer clear, actionable advice for improvement (e.g., "Review the definition of polymorphism," "Remember that ArrayLists provide O(1) access by index," "Check the syntax for declaring arrays").
*   **General:**
    *   **Code Output Questions:** Be precise. The student's output must match the expected output exactly unless the question allows flexibility.
    *   **Conceptual Questions:** Allow for paraphrasing if the core concepts are correct and accurately conveyed.
    *   **Code Writing Questions:** Check for functional correctness and adherence to requirements. Minor stylistic differences are acceptable if the logic is sound.
    *   **Be Lenient:** Ignore minor typos or formatting issues that don't affect meaning or code execution.
    *   **Populate JSON Fully:** Ensure all fields in the requested JSON structure are populated based on your evaluation. Fill lists like `strengths`, `weaknesses`, `suggestions`, `covered`, `missing` appropriately. Provide scores for `accuracy`, `completeness`, `clarity`, and `overall`.

You MUST respond ONLY with the JSON object described below. Do not include any introductory text or explanations outside the JSON structure.
"""
            # Break down the prompt to avoid nested f-strings
            question_part = f"""Evaluate the following student answer based on the question.

Question:
```java
{question_text}
```

Student's Answer:
```java
{student_answer}
```"""

            format_part = """
Respond ONLY with a valid JSON object in this exact format:
{
    "is_correct": true, // boolean: Is the core answer correct?
    "score": 1.0, // float: Overall score (0.0-1.0) based on correctness, completeness, clarity.
    "feedback": "Overall feedback summary based on strengths/weaknesses/suggestions.", // string: A concise summary of the detailed feedback.
    "explanation": "Brief explanation of why the answer is correct/incorrect and how the score was derived.", // string: Justification for the evaluation.
    "validation_details": {
        "concepts": {
            "covered": ["List", "key concepts", "correctly addressed"], // array of strings
            "missing": ["List", "key concepts", "that were missed"] // array of strings
        },
        "scores": {
            "accuracy": 1.0, // float: How factually correct is the answer? (0.0 to 1.0) 
            "completeness": 1.0, // float: Does it cover all parts of the question? (0.0 to 1.0)
            "clarity": 1.0, // float: Is the answer clear and easy to understand? (0.0 to 1.0)
            "specificity": 1.0, // float: How specific and detailed is the answer? (0.0 to 1.0)
            "relevance": 1.0, // float: How relevant is the answer to the question? (0.0 to 1.0)
            "overall": 1.0 // float: Weighted average of all scores (0.0-1.0)
        },
            "feedback_details": {
                "strengths": ["Specific positive aspects", "even if correct"], // array of strings: What did the student do well?
                "weaknesses": ["Specific errors", "or misunderstandings"], // array of strings: What did the student get wrong?
                "suggestions": ["Actionable advice", "for improvement"] // array of strings: How can the student improve?
            }
        }
    }

Scoring Guidelines:
1. Accuracy: 
   - 1.0 = Perfectly correct
   - 0.8-0.9 = Mostly correct with minor inaccuracies
   - 0.6-0.7 = Partially correct with some errors
   - 0.4-0.5 = Mostly incorrect but shows some understanding
   - 0.0-0.3 = Completely wrong

2. Completeness:
   - 1.0 = Covers all key points
   - 0.7-0.9 = Missing minor details
   - 0.4-0.6 = Missing important aspects
   - 0.1-0.3 = Only addresses small part
   - 0.0 = Doesn't address question

3. Clarity:
   - 1.0 = Exceptionally clear and well-organized
   - 0.7-0.9 = Mostly clear with minor issues
   - 0.4-0.6 = Somewhat unclear
   - 0.1-0.3 = Difficult to understand
   - 0.0 = Incomprehensible

4. Specificity:
   - 1.0 = Very specific with excellent examples
   - 0.7-0.9 = Specific with some examples
   - 0.4-0.6 = Somewhat general
   - 0.1-0.3 = Very general
   - 0.0 = Vague

5. Relevance:
   - 1.0 = Perfectly on-topic
   - 0.7-0.9 = Mostly relevant
   - 0.4-0.6 = Somewhat relevant
   - 0.1-0.3 = Barely relevant
   - 0.0 = Off-topic

6. Overall: Weighted average (Accuracy 30%, Completeness 25%, Clarity 20%, Specificity 15%, Relevance 10%)

Examples:
- Perfect answer: All scores 1.0
- Good answer with minor issues: 0.9, 0.8, 0.9, 0.8, 1.0 → ~0.87
- Partial answer: 0.6, 0.5, 0.7, 0.5, 0.8 → ~0.61
- Weak answer: 0.3, 0.2, 0.4, 0.3, 0.5 → ~0.32

IMPORTANT:
1. The entire response must be ONLY the JSON object. No extra text before or after.
2. Ensure all fields are present and have the correct data type.
3. Use double quotes for all keys and string values.
4. Populate all lists (`covered`, `missing`, `strengths`, `weaknesses`, `suggestions`) appropriately based on your evaluation. Provide feedback even for correct answers in 'strengths'.
"""

            user_content = question_part + format_part

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ]
        
        # Increase temperature slightly for more nuanced feedback, but keep it low for consistency
        response = self._call_api(messages, temperature=0.3) 
        
        try:
            response_content = response['choices'][0]['message']['content']
            logger.debug(f"Raw Dissect API response content for check_answer_correctness:\n---\n{response_content}\n---") # Added logging
            # Extract JSON from the response
            import re
            json_match = re.search(r'\{.*\}', response_content, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group(0))
                return result
            else:
                # If no JSON found, try to parse the whole content
                return json.loads(response_content)
        except (KeyError, json.JSONDecodeError) as e:
            logger.error(f"Error parsing Dissect response: {str(e)}")
            logger.error(f"Response content: {response}")
            raise ValueError(f"Invalid response format from Dissect: {e}")
