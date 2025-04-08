# topic_classification/services/dissect_classifier.py

import logging
from typing import Dict, List, Tuple, Any, Optional

from django.utils import timezone
from core.models import Topic, Question
from topic_classification.models import ClassificationModel, ClassificationResult
from topic_classification.services.classifier import TopicClassifier
from .dissect_client import DissectClient

logger = logging.getLogger(__name__)

class DissectClassifier(TopicClassifier):
    """
    Topic classifier using Dissect API.
    """

    def __init__(self, model_id: int):
        super().__init__(model_id)
        self.client = DissectClient()

    def classify_question(self, question: Question) -> Tuple[Topic, float]:
        """
        Classify a question using Dissect API.

        Args:
            question: Question to classify
            
        Returns:
            Tuple containing the topic and confidence score
        """
        # Get all available topics
        available_topics = [
            {
                'id': topic_id,
                'name': self.topics[topic_id]['name'],
                'description': self.topics[topic_id]['description']
            }
            for topic_id in self.topics
        ]
        
        try:
            # Call Dissect API to classify
            result = self.client.classify_topic(question.text, available_topics)

            # Get topic_id and confidence from result
            topic_id = int(result.get('topic_id'))
            confidence = float(result.get('confidence', 0.8))
            
            # Get the topic object
            topic = Topic.objects.get(pk=topic_id)
            
            return topic, confidence

        except Exception as e:
            logger.error(f"Error classifying with Dissect: {str(e)}")

            # Fall back to a simple heuristic
            # Find topic with highest keyword match
            best_match = None
            best_score = 0
            
            for topic_id, topic_data in self.topics.items():
                # Simple keyword matching
                topic_keywords = topic_data['name'].lower().split() + topic_data['description'].lower().split()
                question_words = question.text.lower().split()
                
                # Count matches
                matches = sum(1 for word in question_words if word in topic_keywords)
                score = matches / len(question_words) if question_words else 0
                
                if score > best_score:
                    best_score = score
                    best_match = topic_id
            
            if best_match:
                return Topic.objects.get(pk=best_match), best_score
            else:
                # Get a default topic (first one)
                first_topic = next(iter(self.topics.keys()))
                return Topic.objects.get(pk=first_topic), 0.1
