
import os
import logging
import json
import pickle
from typing import Dict, List, Tuple, Any, Optional, Type

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from django.conf import settings
from django.utils import timezone

from sentence_transformers import SentenceTransformer
from transformers import pipeline

from core.models import Topic, Question
from topic_classification.models import ClassificationModel, ClassificationResult, ClassificationJob

logger = logging.getLogger(__name__)

# Forward declaration
class TopicClassifier:
    """Base class for topic classification."""
    pass


class TopicClassifier:
    """Base class for topic classification."""
    
    def __init__(self, model_id: int):
        """
        Initialize with a ClassificationModel ID.
        """
        self.model = ClassificationModel.objects.get(pk=model_id)
        self.topics = {}
        self._load_topics()
    
    def _load_topics(self):
        """
        Load all topics from the database.
        """
        all_topics = Topic.objects.all()
        self.topics = {topic.id: {
            'name': topic.name,
            'description': topic.description,
            'parent': topic.parent_id
        } for topic in all_topics}
    
    def classify_question(self, question: Question) -> Tuple[Topic, float]:
        """
        Classify a question and return the most likely topic with confidence.
        To be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def save_result(self, job: ClassificationJob, question: Question, 
                   topic: Topic, confidence: float, raw_output: Dict = None) -> ClassificationResult:
        """
        Save classification result to the database.
        """
        if raw_output is None:
            raw_output = {}
            
        result = ClassificationResult.objects.create(
            job=job,
            question=question,
            topic=topic,
            confidence=confidence,
            raw_output=raw_output
        )
        
        # Update the question's topic
        question.topic = topic
        question.save(update_fields=['topic'])
        
        return result
    
    def classify_batch(self, job_id: int) -> bool:
        """
        Process a batch classification job.
        """
        job = ClassificationJob.objects.get(pk=job_id)
        
        try:
            # Update job status
            job.status = 'processing'
            job.started_at = timezone.now()
            job.save(update_fields=['status', 'started_at'])
            
            # Get questions without topics
            questions = Question.objects.filter(topic__isnull=True)
            total_questions = questions.count()
            job.total_questions = total_questions
            job.save(update_fields=['total_questions'])
            
            for i, question in enumerate(questions):
                try:
                    topic, confidence = self.classify_question(question)
                    self.save_result(job, question, topic, confidence)
                    
                    job.classified_questions += 1
                    job.save(update_fields=['classified_questions'])
                    
                    # Log progress every 100 questions
                    if (i + 1) % 100 == 0:
                        logger.info(f"Classified {i + 1}/{total_questions} questions")
                
                except Exception as e:
                    logger.error(f"Error classifying question {question.question_id}: {str(e)}")
                    job.failed_questions += 1
                    job.save(update_fields=['failed_questions'])
            
            # Update job status
            job.status = 'completed'
            job.completed_at = timezone.now()
            job.save(update_fields=['status', 'completed_at'])
            
            return True
            
        except Exception as e:
            error_message = f"Error processing classification job: {str(e)}"
            logger.error(error_message)
            
            # Update job status
            job.status = 'failed'
            job.error_message = error_message
            job.save(update_fields=['status', 'error_message'])
            
            return False


class TfidfClassifier(TopicClassifier):
    """
    Topic classifier using TF-IDF and cosine similarity.
    """
    
    def __init__(self, model_id: int):
        super().__init__(model_id)
        self.vectorizer = None
        self.topic_vectors = None
        self.topic_ids = None
        
        if self.model.model_path and os.path.exists(self.model.model_path):
            self._load_model()
        else:
            self._train_model()
    
    def _load_model(self):
        """
        Load a trained TF-IDF model from disk.
        """
        try:
            with open(self.model.model_path, 'rb') as f:
                model_data = pickle.load(f)
                
            self.vectorizer = model_data['vectorizer']
            self.topic_vectors = model_data['topic_vectors']
            self.topic_ids = model_data['topic_ids']
            
            logger.info(f"Loaded TF-IDF model from {self.model.model_path}")
        except Exception as e:
            logger.error(f"Error loading TF-IDF model: {str(e)}")
            self._train_model()
    
    def _train_model(self):
        """
        Train a new TF-IDF model.
        """
        logger.info("Training new TF-IDF model")
        
        # Get all topics
        topics = Topic.objects.all()
        
        # Create documents from topic names and descriptions
        topic_docs = []
        topic_ids = []
        
        for topic in topics:
            doc = f"{topic.name} {topic.description}"
            topic_docs.append(doc)
            topic_ids.append(topic.id)
        
        # Create and fit the vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        self.topic_vectors = self.vectorizer.fit_transform(topic_docs)
        self.topic_ids = topic_ids
        
        # Save the model
        model_dir = os.path.dirname(self.model.model_path)
        os.makedirs(model_dir, exist_ok=True)
        
        with open(self.model.model_path, 'wb') as f:
            pickle.dump({
                'vectorizer': self.vectorizer,
                'topic_vectors': self.topic_vectors,
                'topic_ids': self.topic_ids
            }, f)
        
        # Update model status
        self.model.status = 'active'
        self.model.save(update_fields=['status'])
        
        logger.info(f"Trained and saved TF-IDF model to {self.model.model_path}")
    
    def classify_question(self, question: Question) -> Tuple[Topic, float]:
        """
        Classify a question using TF-IDF and cosine similarity.
        """
        # Vectorize the question text
        question_vector = self.vectorizer.transform([question.text])
        
        # Calculate similarity to all topics
        similarities = cosine_similarity(question_vector, self.topic_vectors).flatten()
        
        # Find the most similar topic
        top_index = np.argmax(similarities)
        top_similarity = similarities[top_index]
        top_topic_id = self.topic_ids[top_index]
        
        # Get the topic object
        topic = Topic.objects.get(pk=top_topic_id)
        
        return topic, float(top_similarity)


class TransformerClassifier(TopicClassifier):
    """
    Topic classifier using Sentence Transformer embeddings.
    """
    
    def __init__(self, model_id: int):
        super().__init__(model_id)
        self.sentence_model = None
        self.topic_embeddings = None
        self.topic_ids = None
        
        # Load or initialize model
        self._load_transformer()
        
        if self.model.model_path and os.path.exists(self.model.model_path):
            self._load_embeddings()
        else:
            self._generate_embeddings()
    
    def _load_transformer(self):
        """
        Load the sentence transformer model.
        """
        model_name = settings.EMBEDDING_MODEL
        try:
            self.sentence_model = SentenceTransformer(model_name)
            logger.info(f"Loaded Sentence Transformer model: {model_name}")
        except Exception as e:
            logger.error(f"Error loading Sentence Transformer model: {str(e)}")
            raise
    
    def _load_embeddings(self):
        """
        Load pre-computed topic embeddings from disk.
        """
        try:
            with open(self.model.model_path, 'rb') as f:
                data = pickle.load(f)
                
            self.topic_embeddings = data['topic_embeddings']
            self.topic_ids = data['topic_ids']
            
            logger.info(f"Loaded topic embeddings from {self.model.model_path}")
        except Exception as e:
            logger.error(f"Error loading topic embeddings: {str(e)}")
            self._generate_embeddings()
    
    def _generate_embeddings(self):
        """
        Generate embeddings for all topics.
        """
        logger.info("Generating topic embeddings with Sentence Transformer")
        
        # Get all topics
        topics = Topic.objects.all()
        
        # Create documents from topic names and descriptions
        topic_texts = []
        topic_ids = []
        
        for topic in topics:
            text = f"{topic.name}. {topic.description}"
            topic_texts.append(text)
            topic_ids.append(topic.id)
        
        # Generate embeddings
        self.topic_embeddings = self.sentence_model.encode(topic_texts)
        self.topic_ids = topic_ids
        
        # Save the embeddings
        model_dir = os.path.dirname(self.model.model_path)
        os.makedirs(model_dir, exist_ok=True)
        
        with open(self.model.model_path, 'wb') as f:
            pickle.dump({
                'topic_embeddings': self.topic_embeddings,
                'topic_ids': self.topic_ids
            }, f)
        
        # Update model status
        self.model.status = 'active'
        self.model.save(update_fields=['status'])
        
        logger.info(f"Generated and saved topic embeddings to {self.model.model_path}")
    
    def classify_question(self, question: Question) -> Tuple[Topic, float]:
        """
        Classify a question using Sentence Transformer embeddings and cosine similarity.
        """
        # Generate embedding for the question
        question_embedding = self.sentence_model.encode(question.text)
        
        # Calculate similarity to all topics
        similarities = cosine_similarity(
            question_embedding.reshape(1, -1), 
            self.topic_embeddings
        ).flatten()
        
        # Find the most similar topic
        top_index = np.argmax(similarities)
        top_similarity = similarities[top_index]
        top_topic_id = self.topic_ids[top_index]
        
        # Get the topic object
        topic = Topic.objects.get(pk=top_topic_id)
        
        return topic, float(top_similarity)


class APIClassifier(TopicClassifier):
    """
    Topic classifier using external API services (OpenAI, Anthropic, etc.).
    """
    
    def __init__(self, model_id: int):
        super().__init__(model_id)
        self.api_config = self.model.metadata.get('api_config', {})
        self.topics_list = self._get_topics_list()
    
    def _get_topics_list(self) -> List[Dict]:
        """
        Get a structured list of topics for the prompt.
        """
        topics = Topic.objects.all()
        return [
            {
                'id': topic.id,
                'name': topic.name,
                'description': topic.description,
                'parent': topic.parent_id
            }
            for topic in topics
        ]
    
    def _get_prompt(self, question_text: str) -> str:
        """
        Generate a classification prompt for the API.
        """
        base_prompt = """
        You are an expert in Java programming and educational content classification.
        Classify the following Java programming question into the most appropriate topic.
        
        Topics available:
        {topics}
        
        Question to classify:
        {question}
        
        Respond with a JSON object with the following structure:
        {{
            "topic_id": "The ID of the most relevant topic",
            "confidence": "A number between 0 and 1 indicating your confidence",
            "explanation": "Brief explanation of why this topic is the best match"
        }}
        """
        
        # Format topics for the prompt
        topics_formatted = "\n".join([
            f"ID: {t['id']}, Name: {t['name']}, Description: {t['description']}"
            for t in self.topics_list
        ])
        
        return base_prompt.format(
            topics=topics_formatted,
            question=question_text
        )
    
    def _call_openai_api(self, prompt: str) -> Dict:
        """
        Call the OpenAI API for classification.
        """
        try:
            # This is a placeholder for actual API integration
            # In a real implementation, use the openai package
            
            # Example response structure
            response = {
                "topic_id": 1,  # This would come from the API
                "confidence": 0.85,
                "explanation": "This question deals with Java variables and data types."
            }
            
            return response
        except Exception as e:
            logger.error(f"Error calling OpenAI API: {str(e)}")
            raise
    
    def _call_anthropic_api(self, prompt: str) -> Dict:
        """
        Call the Anthropic API for classification.
        """
        try:
            # This is a placeholder for actual API integration
            # In a real implementation, use the anthropic package
            
            # Example response structure
            response = {
                "topic_id": 1,  # This would come from the API
                "confidence": 0.9,
                "explanation": "This question is about Java variables and primitive types."
            }
            
            return response
        except Exception as e:
            logger.error(f"Error calling Anthropic API: {str(e)}")
            raise
    
    def classify_question(self, question: Question) -> Tuple[Topic, float]:
        """
        Classify a question using an external API.
        """
        prompt = self._get_prompt(question.text)
        
        # Choose the appropriate API based on model type
        if self.model.model_type == 'openai':
            api_response = self._call_openai_api(prompt)
        elif self.model.model_type == 'anthropic':
            api_response = self._call_anthropic_api(prompt)
        else:
            raise ValueError(f"Unsupported API type: {self.model.model_type}")
        
        # Extract topic ID and confidence
        topic_id = api_response.get('topic_id')
        confidence = api_response.get('confidence', 0.0)
        
        # Get the topic object
        topic = Topic.objects.get(pk=topic_id)
        
        return topic, confidence
