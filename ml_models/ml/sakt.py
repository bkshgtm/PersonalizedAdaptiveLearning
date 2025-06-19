import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import Dict, List, Tuple, Any, Optional
import os
import json
import math

logger = logging.getLogger(__name__)


class PositionalEncoding(nn.Module):
    """
    Positional encoding for the transformer model.
    """
    
    def __init__(self, d_model: int, max_seq_len: int = 1000):
        """
        Initialize the positional encoding.
        
        Args:
            d_model: Dimensionality of the model
            max_seq_len: Maximum sequence length
        """
        super(PositionalEncoding, self).__init__()
        
        # Create position encoding matrix
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register buffer (not a parameter, but part of the module)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to the input.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model]
            
        Returns:
            Tensor with positional encoding added
        """
        seq_len = x.size(1)
        return x + self.pe[:seq_len, :]


class SAKTModel(nn.Module):
    """
    Implementation of Self-Attentive Knowledge Tracing model.
    
    Reference:
    "A Self-Attentive model for Knowledge Tracing" - Pandey & Karypis, 2019
    https://dl.acm.org/doi/10.1145/3303970.3303974
    """
    
    def __init__(
        self, 
        num_topics: int, 
        d_model: int = 64,
        n_heads: int = 8,
        dropout: float = 0.2,
        max_seq_len: int = 1000
    ):
        """
        Initialize the SAKT model.
        
        Args:
            num_topics: Number of topics in the dataset
            d_model: Dimensionality of the model
            n_heads: Number of attention heads
            dropout: Dropout probability
            max_seq_len: Maximum sequence length
        """
        super(SAKTModel, self).__init__()
        
        # Add 1 for padding (0)
        self.num_topics = num_topics + 1
        self.d_model = d_model
        self.n_heads = n_heads
        self.dropout_rate = dropout
        
        # Embedding layer for exercises (target topics)
        self.exercise_embedding = nn.Embedding(self.num_topics, d_model)
        
        # Embedding layer for interactions (input topics + correctness)
        # For each topic, we have two input states: incorrect (0) and correct (1)
        self.interaction_embedding = nn.Embedding(self.num_topics * 2, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len)
        
        # Multi-head attention
        self.multihead_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        
        # Feed-forward neural network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )
        
        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        
        # Output layer
        self.output_layer = nn.Linear(d_model, 1)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self, 
        input_ids: torch.Tensor,  # Topic IDs from previous interactions
        input_labels: torch.Tensor,  # Correctness for input_ids
        target_ids: torch.Tensor,  # Topic IDs for prediction
        target_labels: Optional[torch.Tensor] = None  # Correctness for target_ids, for training
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            input_ids: Tensor of topic IDs, shape [batch_size, seq_len]
            input_labels: Tensor of correctness for input_ids, shape [batch_size, seq_len]
            target_ids: Tensor of target topic IDs, shape [batch_size, seq_len]
            target_labels: Optional tensor of target correctness, shape [batch_size, seq_len]
            
        Returns:
            Dictionary containing:
            - logits: Tensor of logits, shape [batch_size, seq_len]
            - loss: Loss tensor if target_labels are provided, otherwise None
        """
        batch_size, seq_len = input_ids.shape
        
        # Validate input_ids are within bounds (0 to num_topics-1)
        input_ids = input_ids.clamp(0, self.num_topics - 1)
        
        # Combine input_ids with correctness
        # (topic_id * 2) for incorrect, (topic_id * 2 + 1) for correct
        if input_labels is not None:
            # Ensure labels are binary (0 or 1)
            input_labels = input_labels.clamp(0, 1).long()
            input_combined = (input_ids * 2 + input_labels).long()
            # Ensure combined values are within bounds
            input_combined = input_combined.clamp(0, self.num_topics * 2 - 1)
        else:
            # If no labels provided, assume all incorrect (even indices)
            input_combined = (input_ids * 2).long()
            input_combined = input_combined.clamp(0, self.num_topics * 2 - 1)
        
        # Embed the inputs
        input_embedded = self.interaction_embedding(input_combined)
        
        # Add positional encoding
        input_embedded = self.pos_encoder(input_embedded)
        
        # Apply dropout
        input_embedded = self.dropout(input_embedded)
        
        # Embed the target exercises
        target_embedded = self.exercise_embedding(target_ids)
        
        # Transpose for attention: [batch_size, seq_len, d_model] -> [seq_len, batch_size, d_model]
        input_embedded = input_embedded.transpose(0, 1)
        target_embedded = target_embedded.transpose(0, 1)
        
        # Self-attention
        attn_output, _ = self.multihead_attn(
            query=target_embedded,
            key=input_embedded,
            value=input_embedded
        )
        
        # Residual connection and layer normalization
        attn_output = target_embedded + self.dropout(attn_output)
        attn_output = self.layer_norm1(attn_output)
        
        # Feed-forward network
        ffn_output = self.ffn(attn_output)
        
        # Another residual connection and layer normalization
        ffn_output = attn_output + self.dropout(ffn_output)
        ffn_output = self.layer_norm2(ffn_output)
        
        # Output layer
        logits = self.output_layer(ffn_output).squeeze(-1)
        
        # Transpose back: [seq_len, batch_size] -> [batch_size, seq_len]
        logits = logits.transpose(0, 1)
        
        # Calculate loss if target_labels are provided
        loss = None
        if target_labels is not None:
            # Use binary cross-entropy loss
            loss = F.binary_cross_entropy_with_logits(
                logits, 
                target_labels.float()
            )
        
        return {
            'logits': logits,
            'loss': loss
        }
    
    def predict(
        self, 
        input_ids: torch.Tensor,
        input_labels: Optional[torch.Tensor],
        topic_ids: List[int]
    ) -> Dict[int, float]:
        """
        Predict the probability of correctness for all topics.
        
        Args:
            input_ids: Tensor of topic IDs, shape [seq_len]
            input_labels: Tensor of correctness labels, shape [seq_len], or None
            topic_ids: List of topic IDs to predict for
            
        Returns:
            Dictionary mapping topic IDs to predicted probabilities
        """
        self.eval()
        
        with torch.no_grad():
            # Add batch dimension
            input_ids = input_ids.unsqueeze(0)
            
            # Clamp input_ids to valid range
            input_ids = input_ids.clamp(0, self.num_topics - 1)
            
            # Combine input_ids with correctness
            if input_labels is not None:
                input_labels = input_labels.unsqueeze(0).clamp(0, 1)
                input_combined = (input_ids * 2 + input_labels.long()).long()
            else:
                # If no labels provided, assume all incorrect (even indices)
                input_combined = (input_ids * 2).long()
            
            # Ensure combined values are within bounds
            input_combined = input_combined.clamp(0, self.num_topics * 2 - 1)
            
            # Create predictions for all topics
            result = {}
            
            for topic_id in topic_ids:
                # Direct mapping: topic_id corresponds to index in embedding
                # topic_id 1 -> index 1, topic_id 2 -> index 2, etc.
                if 1 <= topic_id < self.num_topics:
                    target_tensor = torch.tensor([[topic_id]], dtype=torch.long)
                    
                    # Forward pass
                    outputs = self.forward(
                        input_ids=input_ids,
                        input_labels=input_labels,
                        target_ids=target_tensor
                    )
                    
                    # Get probability for this topic
                    prob = torch.sigmoid(outputs['logits']).item()
                    result[topic_id] = float(prob)
                else:
                    # Out of range, use default
                    result[topic_id] = 0.5
        
        return result
    
    def save(self, model_path: str, topic_ids: List[int], hyperparameters: Dict[str, Any]):
        """
        Save the model and metadata.
        
        Args:
            model_path: Path to save the model
            topic_ids: List of topic IDs
            hyperparameters: Dictionary of hyperparameters
        """
        # Create directory if it doesn't exist
        model_dir = os.path.dirname(model_path)
        os.makedirs(model_dir, exist_ok=True)
        
        # Save the model state dict
        torch.save(self.state_dict(), model_path)
        
        # Save metadata
        metadata_path = os.path.join(model_dir, 'metadata.json')
        metadata = {
            'num_topics': self.num_topics - 1,  # Subtract 1 for padding
            'd_model': self.d_model,
            'n_heads': self.n_heads,
            'dropout': self.dropout_rate,
            'topic_ids': topic_ids,
            'hyperparameters': hyperparameters
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
    
    @classmethod
    def load(cls, model_path: str) -> Tuple['SAKTModel', List[int], Dict[str, Any]]:
        """
        Load a model from disk.
        
        Args:
            model_path: Path to load the model from
            
        Returns:
            Tuple containing:
            - Loaded model
            - List of topic IDs
            - Dictionary of hyperparameters
        """
        # Load metadata
        model_dir = os.path.dirname(model_path)
        metadata_path = os.path.join(model_dir, 'metadata.json')
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Create model
        model = cls(
            num_topics=metadata['num_topics'],
            d_model=metadata['d_model'],
            n_heads=metadata['n_heads'],
            dropout=metadata['dropout']
        )
        
        # Load state dict
        model.load_state_dict(torch.load(model_path))
        
        return model, metadata['topic_ids'], metadata['hyperparameters']


class SAKTTrainer:
    """
    Trainer for the SAKT model.
    """
    
    def __init__(
        self,
        model: SAKTModel,
        topic_ids: List[int],
        device: str = 'cpu',
        learning_rate: float = 0.001
    ):
        """
        Initialize the trainer.
        
        Args:
            model: SAKT model
            topic_ids: List of topic IDs
            device: Device to use for training
            learning_rate: Learning rate for optimizer
        """
        self.model = model
        self.topic_ids = topic_ids
        self.device = device
        
        # Move model to device
        self.model.to(device)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    def train_epoch(self, train_loader):
        """
        Train for one epoch.
        
        Args:
            train_loader: DataLoader for training
            
        Returns:
            Average loss for the epoch
        """
        self.model.train()
        epoch_loss = 0.0
        
        for batch in train_loader:
            # Get batch data
            input_ids = batch['input_ids'].to(self.device)
            target_ids = batch['target_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            
            # For SAKT, the target_ids are the exercise IDs we want to predict
            # The input_ids are the previous exercises
            # We need to create input_labels by shifting the labels
            
            # Create input labels (shift labels to right, prepend a 0)
            input_labels = torch.zeros_like(labels)
            input_labels[:, 1:] = labels[:, :-1]
            
            # Apply attention mask to labels
            labels = labels * attention_mask
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(
                input_ids=input_ids,
                input_labels=input_labels,
                target_ids=target_ids,
                target_labels=labels
            )
            loss = outputs['loss']
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            epoch_loss += loss.item()
        
        return epoch_loss / len(train_loader)
    
    def evaluate(self, test_loader):
        """
        Evaluate the model.
        
        Args:
            test_loader: DataLoader for testing
            
        Returns:
            Dictionary containing:
            - loss: Average loss
            - accuracy: Average accuracy
            - auc: Average AUC
        """
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in test_loader:
                # Get batch data
                input_ids = batch['input_ids'].to(self.device)
                target_ids = batch['target_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                # Create input labels (shift labels to right, prepend a 0)
                input_labels = torch.zeros_like(labels)
                input_labels[:, 1:] = labels[:, :-1]
                
                # Apply attention mask to labels
                labels = labels * attention_mask
                
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    input_labels=input_labels,
                    target_ids=target_ids,
                    target_labels=labels
                )
                loss = outputs['loss']
                
                # Get predictions
                logits = outputs['logits']
                probs = torch.sigmoid(logits)
                
                # Add to lists for metrics calculation
                all_preds.extend(probs.cpu().numpy().flatten())
                all_labels.extend(labels.cpu().numpy().flatten())
                
                total_loss += loss.item()
        
        # Calculate metrics
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        # Calculate accuracy
        predictions = (all_preds >= 0.5).astype(int)
        accuracy = (predictions == all_labels).mean()
        
        # Calculate AUC
        try:
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(all_labels, all_preds)
        except:
            auc = 0.0
        
        return {
            'loss': total_loss / len(test_loader),
            'accuracy': accuracy,
            'auc': auc
        }
    
    def train(self, train_loader, test_loader, num_epochs: int = 10):
        """
        Train the model.
        
        Args:
            train_loader: DataLoader for training
            test_loader: DataLoader for testing
            num_epochs: Number of epochs to train for
            
        Returns:
            Dictionary containing training history
        """
        history = {
            'train_loss': [],
            'test_loss': [],
            'test_accuracy': [],
            'test_auc': []
        }
        
        for epoch in range(num_epochs):
            # Train
            train_loss = self.train_epoch(train_loader)
            
            # Evaluate
            eval_metrics = self.evaluate(test_loader)
            
            # Add to history
            history['train_loss'].append(train_loss)
            history['test_loss'].append(eval_metrics['loss'])
            history['test_accuracy'].append(eval_metrics['accuracy'])
            history['test_auc'].append(eval_metrics['auc'])
            
            # Print metrics
            logger.info(
                f"Epoch {epoch+1}/{num_epochs}: "
                f"train_loss={train_loss:.4f}, "
                f"test_loss={eval_metrics['loss']:.4f}, "
                f"test_acc={eval_metrics['accuracy']:.4f}, "
                f"test_auc={eval_metrics['auc']:.4f}"
            )
        
        return history
