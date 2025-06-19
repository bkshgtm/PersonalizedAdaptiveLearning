import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, List, Tuple, Any
from django.db.models import Q, Count
from django.db import models
from core.models import Student, Topic, StudentInteraction, KnowledgeState
from knowledge_graph.models import KnowledgeGraph
import logging

logger = logging.getLogger(__name__)


class StudentInteractionDataset(Dataset):
    """
    Dataset for student interactions compatible with DKT and SAKT models.
    """
    
    def __init__(self, student_interactions: List[Dict], topic_to_id: Dict[str, int], 
                 max_seq_len: int = 100):
        """
        Initialize the dataset.
        
        Args:
            student_interactions: List of student interaction sequences
            topic_to_id: Mapping from topic names to IDs
            max_seq_len: Maximum sequence length
        """
        self.interactions = student_interactions
        self.topic_to_id = topic_to_id
        self.max_seq_len = max_seq_len
        
    def __len__(self):
        return len(self.interactions)
    
    def __getitem__(self, idx):
        sequence = self.interactions[idx]
        
        # Convert to tensors
        input_ids = []
        labels = []
        
        for interaction in sequence:
            topic_name = interaction['topic']
            topic_id = self.topic_to_id.get(topic_name, 0)  # 0 for unknown topics
            input_ids.append(topic_id)
            labels.append(1 if interaction['correct'] else 0)
        
        # Pad or truncate sequences
        if len(input_ids) > self.max_seq_len:
            input_ids = input_ids[:self.max_seq_len]
            labels = labels[:self.max_seq_len]
        else:
            # Pad with zeros
            padding_length = self.max_seq_len - len(input_ids)
            input_ids.extend([0] * padding_length)
            labels.extend([0] * padding_length)
        
        # Create target_ids (shift input_ids by 1 for next-step prediction)
        target_ids = input_ids[1:] + [0]  # Shift left and pad with 0
        
        # Create attention mask
        attention_mask = [1 if i < len(sequence) else 0 for i in range(self.max_seq_len)]
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'target_ids': torch.tensor(target_ids, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.float),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.float)
        }


class DjangoDataPreparation:
    """
    Data preparation class that works with Django models.
    """
    
    def __init__(self):
        self.topic_to_id = {}
        self.id_to_topic = {}
        self._build_topic_mappings()
    
    def _build_topic_mappings(self):
        """Build mappings between topic names and IDs."""
        topics = Topic.objects.all().order_by('id')
        
        # Reserve 0 for padding/unknown
        self.topic_to_id = {'<PAD>': 0}
        self.id_to_topic = {0: '<PAD>'}
        
        for i, topic in enumerate(topics, 1):
            self.topic_to_id[topic.name] = i
            self.id_to_topic[i] = topic.name
        
        logger.info(f"Built topic mappings for {len(topics)} topics")
    
    def get_student_interaction_sequences(self, min_interactions: int = 5) -> List[Dict]:
        """
        Get student interaction sequences from the database.
        
        Args:
            min_interactions: Minimum number of interactions per student
            
        Returns:
            List of student interaction sequences
        """
        sequences = []
        
        # Get all students with sufficient interactions
        students = Student.objects.annotate(
            interaction_count=models.Count('interactions')
        ).filter(interaction_count__gte=min_interactions)
        
        for student in students:
            # Get interactions ordered by timestamp
            interactions = StudentInteraction.objects.filter(
                student=student
            ).select_related('question', 'question__topic').order_by('timestamp')
            
            sequence = []
            for interaction in interactions:
                if interaction.question.topic:
                    sequence.append({
                        'student_id': student.student_id,
                        'topic': interaction.question.topic.name,
                        'correct': interaction.correct,
                        'timestamp': interaction.timestamp,
                        'time_taken': interaction.time_taken.total_seconds() if interaction.time_taken else 0,
                        'attempt_number': interaction.attempt_number
                    })
            
            if len(sequence) >= min_interactions:
                sequences.append(sequence)
        
        logger.info(f"Prepared {len(sequences)} student sequences")
        return sequences
    
    def create_dataloaders(self, test_split: float = 0.2, batch_size: int = 32, 
                          max_seq_len: int = 100) -> Tuple[DataLoader, DataLoader]:
        """
        Create train and test dataloaders.
        
        Args:
            test_split: Fraction of data to use for testing
            batch_size: Batch size for dataloaders
            max_seq_len: Maximum sequence length
            
        Returns:
            Tuple of (train_loader, test_loader)
        """
        # Get interaction sequences
        sequences = self.get_student_interaction_sequences()
        
        # Split into train and test
        split_idx = int(len(sequences) * (1 - test_split))
        train_sequences = sequences[:split_idx]
        test_sequences = sequences[split_idx:]
        
        # Create datasets
        train_dataset = StudentInteractionDataset(
            train_sequences, self.topic_to_id, max_seq_len
        )
        test_dataset = StudentInteractionDataset(
            test_sequences, self.topic_to_id, max_seq_len
        )
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False
        )
        
        logger.info(f"Created dataloaders: train={len(train_dataset)}, test={len(test_dataset)}")
        return train_loader, test_loader
    
    def get_topic_list(self) -> List[str]:
        """Get list of all topic names."""
        return [self.id_to_topic[i] for i in sorted(self.id_to_topic.keys()) if i > 0]
    
    def get_num_topics(self) -> int:
        """Get number of topics (excluding padding)."""
        return len(self.topic_to_id) - 1
    
    def get_student_current_state(self, student_id: str) -> Dict[str, Any]:
        """
        Get current knowledge state for a student.
        
        Args:
            student_id: Student ID
            
        Returns:
            Dictionary containing student's current state
        """
        try:
            student = Student.objects.get(student_id=student_id)
        except Student.DoesNotExist:
            logger.error(f"Student {student_id} not found")
            return {}
        
        # Get recent interactions (last 50)
        recent_interactions = StudentInteraction.objects.filter(
            student=student
        ).select_related('question', 'question__topic').order_by('-timestamp')[:50]
        
        # Build interaction sequence
        interaction_sequence = []
        topic_performance = {}
        
        for interaction in reversed(recent_interactions):  # Reverse to get chronological order
            if interaction.question.topic:
                topic_name = interaction.question.topic.name
                topic_id = self.topic_to_id.get(topic_name, 0)
                
                interaction_sequence.append({
                    'topic_id': topic_id,
                    'topic_name': topic_name,
                    'correct': interaction.correct,
                    'timestamp': interaction.timestamp
                })
                
                # Track performance per topic
                if topic_name not in topic_performance:
                    topic_performance[topic_name] = {'correct': 0, 'total': 0}
                
                topic_performance[topic_name]['total'] += 1
                if interaction.correct:
                    topic_performance[topic_name]['correct'] += 1
        
        # Calculate current mastery scores
        mastery_scores = {}
        for topic_name, perf in topic_performance.items():
            mastery_scores[topic_name] = perf['correct'] / perf['total'] if perf['total'] > 0 else 0.0
        
        return {
            'student_id': student_id,
            'interaction_sequence': interaction_sequence,
            'topic_performance': topic_performance,
            'mastery_scores': mastery_scores,
            'total_interactions': len(interaction_sequence)
        }
    
    def update_knowledge_states(self, student_id: str, predictions: Dict[str, float]):
        """
        Update KnowledgeState records with new predictions.
        
        Args:
            student_id: Student ID
            predictions: Dictionary mapping topic names to mastery predictions
        """
        try:
            student = Student.objects.get(student_id=student_id)
        except Student.DoesNotExist:
            logger.error(f"Student {student_id} not found")
            return
        
        for topic_name, mastery_score in predictions.items():
            try:
                topic = Topic.objects.get(name=topic_name)
                
                # Update or create knowledge state
                knowledge_state, created = KnowledgeState.objects.update_or_create(
                    student=student,
                    topic=topic,
                    defaults={
                        'proficiency_score': mastery_score,
                        'confidence': 0.8  # Default confidence, can be improved
                    }
                )
                
                if created:
                    logger.info(f"Created knowledge state for {student_id} - {topic_name}: {mastery_score:.3f}")
                else:
                    logger.info(f"Updated knowledge state for {student_id} - {topic_name}: {mastery_score:.3f}")
                    
            except Topic.DoesNotExist:
                logger.warning(f"Topic {topic_name} not found in database")
