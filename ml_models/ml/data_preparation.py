import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Any, Optional
import torch
from torch.utils.data import Dataset, DataLoader

from django.db.models import Q, Count, Prefetch

from core.models import Student, Topic, StudentInteraction, Question

logger = logging.getLogger(__name__)


class InteractionSequenceDataset(Dataset):
    """
    Dataset for student interaction sequences.
    """
    
    def __init__(self, interactions: List[Dict], topic_dict: Dict[int, int], max_seq_len: int = 64):
        """
        Initialize the dataset.
        
        Args:
            interactions: List of dictionaries with student interaction data
            topic_dict: Dictionary mapping topic IDs to indices
            max_seq_len: Maximum sequence length
        """
        self.interactions = interactions
        self.topic_dict = topic_dict
        self.max_seq_len = max_seq_len
        self.num_topics = len(topic_dict)
        
        # Pre-process the data
        self.student_sequences = self._preprocess_data()
        
    def _preprocess_data(self) -> List[Dict]:
        """
        Group interactions by student into sequences.
        
        Returns:
            List of dictionaries with student sequences
        """
        sequences = {}
        
        logger.debug(f"Processing {len(self.interactions)} interactions")
        
        for i, interaction in enumerate(self.interactions):
            student_id = interaction['student_id']
            topic_id = interaction['topic_id']
            
            if student_id not in sequences:
                sequences[student_id] = {
                    'topic_ids': [],
                    'correctness': [],
                    'timestamps': []
                }
                logger.debug(f"New student sequence: {student_id}")
            
            # Validate interaction data
            if not all(k in interaction for k in ['student_id', 'topic_id', 'correct', 'timestamp']):
                logger.error(f"Invalid interaction at index {i}: {interaction}")
                continue
            
            # Map topic ID to index
            topic_idx = self.topic_dict.get(topic_id, 0)  # Default to 0 if topic not found
            
            sequences[student_id]['topic_ids'].append(topic_idx)
            sequences[student_id]['correctness'].append(1 if interaction['correct'] else 0)
            sequences[student_id]['timestamps'].append(interaction['timestamp'])
        
        # Sort sequences by timestamp and convert to list
        result = []
        for student_id, data in sequences.items():
            # Sort by timestamp
            sorted_indices = np.argsort(data['timestamps'])
            
            # Apply sorting
            seq = {
                'student_id': student_id,
                'topic_ids': [data['topic_ids'][i] for i in sorted_indices],
                'correctness': [data['correctness'][i] for i in sorted_indices]
            }
            
            # Add to result
            result.append(seq)
            
        return result
    
    def __len__(self):
        return len(self.student_sequences)
    
    def _empty_sequence(self):
        """
        Return an empty sequence for error cases
        """
        return {
            'input_ids': torch.LongTensor([]),
            'target_ids': torch.LongTensor([]),
            'labels': torch.FloatTensor([]),
            'seq_len': 0
        }
        
    def __getitem__(self, idx):
        """
        Get a student sequence (without padding)
        
        Args:
            idx: Index of the student sequence
        
        Returns:
            Dictionary with raw sequences
        """
        seq = self.student_sequences[idx]
        topic_ids = seq['topic_ids']
        correctness = seq['correctness']
        
        # Validate sequence
        if len(topic_ids) != len(correctness):
            logger.error(f"Sequence {idx} has mismatched lengths: {len(topic_ids)} topics vs {len(correctness)} correctness")
            return self._empty_sequence()
        
        # Truncate if longer than max length
        if len(topic_ids) > self.max_seq_len:
            topic_ids = topic_ids[-self.max_seq_len:]
            correctness = correctness[-self.max_seq_len:]
        
        # Validate sequence length
        if len(topic_ids) < 2:
            logger.warning(f"Sequence too short ({len(topic_ids)}), skipping")
            return self._empty_sequence()
        if len(topic_ids) > self.max_seq_len:
            logger.warning(f"Sequence too long ({len(topic_ids)}), truncating to {self.max_seq_len}")
            
        # Input is all but last, target is all but first
        input_ids = topic_ids[:-1]
        target_ids = topic_ids[1:]
        labels = correctness[1:]
        
        return {
            'input_ids': torch.LongTensor(input_ids),
            'target_ids': torch.LongTensor(target_ids),
            'labels': torch.FloatTensor(labels),
            'seq_len': len(input_ids)
        }

    @staticmethod
    def collate_fn(batch):
        """
        Custom collate function to handle variable length sequences
        with strict size validation and error recovery
        """
        # Filter out empty sequences
        batch = [b for b in batch if b['seq_len'] > 0]
        
        if not batch:
            return None
            
        # Validate all sequences have matching lengths
        for i, b in enumerate(batch):
            if len(b['input_ids']) != b['seq_len'] or \
               len(b['target_ids']) != b['seq_len'] or \
               len(b['labels']) != b['seq_len']:
                logger.error(f"Sequence {i} has mismatched lengths: "
                           f"input={len(b['input_ids'])}, "
                           f"target={len(b['target_ids'])}, "
                           f"labels={len(b['labels'])}, "
                           f"expected={b['seq_len']}")
                continue
            
        # Get max sequence length in this batch
        max_len = max(b['seq_len'] for b in batch)
        
        # Pad all sequences to max length
        padded_batch = []
        for b in batch:
            try:
                pad_len = max_len - b['seq_len']
                
                # Validate tensor sizes before padding
                assert len(b['input_ids']) == b['seq_len']
                assert len(b['target_ids']) == b['seq_len']
                assert len(b['labels']) == b['seq_len']
                
                padded_input = torch.nn.functional.pad(
                    b['input_ids'], 
                    (0, pad_len), 
                    value=0
                )
                padded_target = torch.nn.functional.pad(
                    b['target_ids'],
                    (0, pad_len),
                    value=0
                )
                padded_labels = torch.nn.functional.pad(
                    b['labels'],
                    (0, pad_len),
                    value=0.0
                )
                attention_mask = torch.cat([
                    torch.ones(b['seq_len'], dtype=torch.long),
                    torch.zeros(pad_len, dtype=torch.long)
                ])
                
                padded_batch.append({
                    'input_ids': padded_input,
                    'target_ids': padded_target,
                    'labels': padded_labels,
                    'attention_mask': attention_mask,
                    'seq_len': b['seq_len']
                })
                
                logger.debug(f"Padded sequence: original_len={b['seq_len']}, padded_len={max_len}")
                
            except Exception as e:
                logger.error(f"Failed to pad sequence: {str(e)}")
                continue
            
        # Stack all tensors
        return {
            'input_ids': torch.stack([b['input_ids'] for b in padded_batch]),
            'target_ids': torch.stack([b['target_ids'] for b in padded_batch]),
            'labels': torch.stack([b['labels'] for b in padded_batch]),
            'attention_mask': torch.stack([b['attention_mask'] for b in padded_batch]),
            'seq_lens': torch.LongTensor([b['seq_len'] for b in padded_batch])
        }


class DataPreparation:
    """
    Service for preparing data for knowledge tracing models.
    """
    
    def __init__(self, course_id: str):
        """
        Initialize with a course ID.
        
        Args:
            course_id: ID of the course
        """
        self.course_id = course_id
    
    def get_topic_mapping(self) -> Tuple[Dict[int, int], List[int]]:
        """
        Get a mapping from topic IDs to indices.
        
        Returns:
            Tuple containing:
            - Dictionary mapping topic IDs to indices
            - List of topic IDs
        """
        # Get all topics for the course
        topics = Topic.objects.filter(course__course_id=self.course_id)
        
        # Create the mapping
        topic_dict = {topic.id: idx + 1 for idx, topic in enumerate(topics)}  # Start from 1, 0 reserved for padding
        topic_ids = [topic.id for topic in topics]
        
        return topic_dict, topic_ids
    
    def get_interactions(self, students: Optional[List[str]] = None) -> List[Dict]:
        """
        Get all student interactions for the course.
        
        Args:
            students: Optional list of student IDs to filter by
        
        Returns:
            List of dictionaries with interaction data
        """
        # Build the query
        query = Q(question__assessment__course__course_id=self.course_id)
        
        if students:
            query &= Q(student__student_id__in=students)
        
        # Get only interactions with questions that have topics
        query &= Q(question__topic__isnull=False)
        
        # Get the interactions
        interactions = StudentInteraction.objects.filter(query).select_related(
            'student', 'question', 'question__topic'
        ).order_by('student', 'timestamp')
        
        # Convert to list of dictionaries
        result = []
        for interaction in interactions:
            result.append({
                'student_id': interaction.student.student_id,
                'question_id': interaction.question.question_id,
                'topic_id': interaction.question.topic.id,
                'correct': interaction.correct,
                'timestamp': interaction.timestamp.timestamp()  # Convert to Unix timestamp
            })
        
        return result
    
    def split_data(self, interactions: List[Dict], test_ratio: float = 0.2) -> Tuple[List[Dict], List[Dict]]:
        """
        Split interactions into training and testing sets.
        
        Args:
            interactions: List of dictionaries with interaction data
            test_ratio: Ratio of data to use for testing
        
        Returns:
            Tuple containing:
            - List of training interactions
            - List of testing interactions
        """
        # Get all unique student IDs
        student_ids = list(set(interaction['student_id'] for interaction in interactions))
        
        # Shuffle the student IDs
        np.random.shuffle(student_ids)
        
        # Split the student IDs
        split_idx = int(len(student_ids) * (1 - test_ratio))
        train_students = student_ids[:split_idx]
        test_students = student_ids[split_idx:]
        
        # Split the interactions
        train_interactions = [
            interaction for interaction in interactions
            if interaction['student_id'] in train_students
        ]
        
        test_interactions = [
            interaction for interaction in interactions
            if interaction['student_id'] in test_students
        ]
        
        return train_interactions, test_interactions
    
    def create_dataloaders(
        self, 
        train_interactions: List[Dict], 
        test_interactions: List[Dict],
        topic_dict: Dict[int, int],
        batch_size: int = 32,
        max_seq_len: int = 200
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Create DataLoader objects for training and testing.
        
        Args:
            train_interactions: List of training interactions
            test_interactions: List of testing interactions
            topic_dict: Dictionary mapping topic IDs to indices
            batch_size: Batch size for DataLoader
            max_seq_len: Maximum sequence length
        
        Returns:
            Tuple containing:
            - Training DataLoader
            - Testing DataLoader
        """
        # Create datasets
        train_dataset = InteractionSequenceDataset(train_interactions, topic_dict, max_seq_len)
        test_dataset = InteractionSequenceDataset(test_interactions, topic_dict, max_seq_len)
        
        # Create dataloaders with custom collate function
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=InteractionSequenceDataset.collate_fn
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=InteractionSequenceDataset.collate_fn
        )
        
        return train_loader, test_loader
    
    def prepare_for_prediction(
        self, 
        student_id: str,
        topic_dict: Dict[int, int],
    ) -> Dict:
        """
        Prepare data for a single student prediction.
        
        Args:
            student_id: ID of the student
            topic_dict: Dictionary mapping topic IDs to indices
        
        Returns:
            Dictionary with input data for prediction
        """
        # Get all interactions for the student
        interactions = self.get_interactions(students=[student_id])
        
        # Filter interactions to only include topics in the training set
        filtered_interactions = [
            interaction for interaction in interactions
            if interaction['topic_id'] in topic_dict
        ]
        
        # Create dataset with filtered interactions
        dataset = InteractionSequenceDataset(filtered_interactions, topic_dict)
        
        # There should be only one student
        if len(dataset) == 0:
            return None
        
        return dataset[0]
    
    def get_topic_performance(self, student_id: str) -> Dict[int, Dict]:
        """
        Get a student's performance by topic.
        
        Args:
            student_id: ID of the student
        
        Returns:
            Dictionary mapping topic IDs to performance metrics
        """
        # Get all interactions for the student
        query = Q(student__student_id=student_id)
        query &= Q(question__assessment__course__course_id=self.course_id)
        query &= Q(question__topic__isnull=False)
        
        interactions = StudentInteraction.objects.filter(query).select_related(
            'question', 'question__topic'
        )
        
        # Group by topic
        topic_performance = {}
        
        for interaction in interactions:
            topic_id = interaction.question.topic.id
            
            if topic_id not in topic_performance:
                topic_performance[topic_id] = {
                    'total': 0,
                    'correct': 0,
                    'last_timestamp': None
                }
            
            topic_performance[topic_id]['total'] += 1
            
            if interaction.correct:
                topic_performance[topic_id]['correct'] += 1
            
            # Update last timestamp
            last_ts = topic_performance[topic_id]['last_timestamp']
            if last_ts is None or interaction.timestamp > last_ts:
                topic_performance[topic_id]['last_timestamp'] = interaction.timestamp
        
        # Calculate accuracy
        for topic_id, data in topic_performance.items():
            data['accuracy'] = data['correct'] / data['total'] if data['total'] > 0 else 0
        
        return topic_performance
