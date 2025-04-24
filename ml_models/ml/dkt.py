import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import Dict, List, Tuple, Any, Optional
import os
import json

logger = logging.getLogger(__name__)


class DKTModel(nn.Module):
    """
    Implementation of Deep Knowledge Tracing model.
    
    Reference: 
    "Deep Knowledge Tracing" - Piech et al., 2015
    https://papers.nips.cc/paper/5654-deep-knowledge-tracing.pdf
    """
    
    def __init__(
        self, 
        num_topics: int, 
        hidden_size: int = 100,
        num_layers: int = 1,
        dropout: float = 0.2
    ):
        """
        Initialize the DKT model.
        
        Args:
            num_topics: Number of topics in the dataset
            hidden_size: Hidden size of the LSTM
            num_layers: Number of LSTM layers
            dropout: Dropout probability
        """
        super(DKTModel, self).__init__()
        
        # Add 1 for padding (0)
        self.num_topics = num_topics + 1
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Input size: one-hot encoding of the topic + correctness
        # For each topic, we have two inputs: (topic, 0) and (topic, 1)
        # This creates a combined embedding for topic and correctness
        self.input_size = self.num_topics * 2
        
        # Embedding layer maps each topic+correctness pair to a dense vector
        self.embedding = nn.Embedding(self.input_size, hidden_size)
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Output layer: predict correctness for each topic
        self.output_layer = nn.Linear(hidden_size, self.num_topics)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self, 
        input_ids: torch.Tensor,  # Topic IDs from previous interactions
        target_ids: torch.Tensor,  # Topic IDs for prediction
        labels: Optional[torch.Tensor] = None  # Correctness for target_ids
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            input_ids: Tensor of topic IDs, shape [batch_size, seq_len]
            target_ids: Tensor of target topic IDs for prediction, shape [batch_size, seq_len]
            labels: Optional tensor of correctness labels, shape [batch_size, seq_len]
            
        Returns:
            Dictionary containing:
            - logits: Tensor of logits for each topic, shape [batch_size, seq_len, num_topics]
            - loss: Loss tensor if labels are provided, otherwise None
        """
        batch_size, seq_len = input_ids.shape
        
        # Convert input_ids to one-hot encoding
        input_combined = input_ids.clone()
        
        # If we have labels for the input sequence, combine topic_id with correctness
        if labels is not None:
            # For each topic ID, we create two new IDs:
            # - topic_id * 2: incorrect answer
            # - topic_id * 2 + 1: correct answer
            input_combined = (input_ids * 2 + labels).long()
            
        # Embed the input
        embedded = self.embedding(input_combined.long())
        embedded = self.dropout(embedded)
        
        # Pass through LSTM
        lstm_out, _ = self.lstm(embedded)
        lstm_out = self.dropout(lstm_out)
        
        # Get output logits
        logits = self.output_layer(lstm_out)
        
        # Calculate loss if labels are provided
        loss = None
        if labels is not None:
            # For each position, get the logit for the target topic
            # We need to gather the logits for the target topics
            target_logits = torch.gather(
                logits, 
                dim=2, 
                index=target_ids.unsqueeze(2)
            ).squeeze(2)
            
            # Calculate binary cross-entropy loss
            loss = F.binary_cross_entropy_with_logits(
                target_logits, 
                labels.float()
            )
        
        return {
            'logits': logits,
            'loss': loss
        }
    
    def predict(
        self, 
        input_ids: torch.Tensor,
        input_labels: torch.Tensor,
        topic_ids: List[int]
    ) -> Dict[int, float]:
        """
        Predict the probability of correctness for all topics.
        
        Args:
            input_ids: Tensor of topic IDs, shape [seq_len]
            input_labels: Tensor of correctness labels, shape [seq_len]
            topic_ids: List of topic IDs to predict for
            
        Returns:
            Dictionary mapping topic IDs to predicted probabilities
        """
        # Make sure we're in evaluation mode
        self.eval()
        
        # Add batch dimension
        input_ids = input_ids.unsqueeze(0)
        input_labels = input_labels.unsqueeze(0)
        
        # Combine topic IDs with correctness
        if input_labels is not None:
            input_combined = (input_ids * 2 + input_labels).long()
        else:
            # If no labels provided, assume all incorrect (even indices)
            input_combined = (input_ids * 2).long()
        
        # Embed the input
        embedded = self.embedding(input_combined)
        
        # Pass through LSTM
        with torch.no_grad():
            lstm_out, _ = self.lstm(embedded)
            
            # Get the last hidden state
            last_hidden = lstm_out[:, -1, :]
            
            # Get output logits for all topics
            all_logits = self.output_layer(last_hidden)
            
            # Apply sigmoid to get probabilities
            all_probs = torch.sigmoid(all_logits).squeeze(0).cpu().numpy()
        
        # Create a dictionary mapping topic IDs to probabilities
        result = {}
        for topic_id in topic_ids:
            # Map the topic_id to its index in the embedding
            # Add 1 because 0 is reserved for padding
            topic_idx = 0
            for i, tid in enumerate(topic_ids):
                if tid == topic_id:
                    topic_idx = i + 1
                    break
            
            if topic_idx == 0 or topic_idx >= len(all_probs):
                # Topic not found in the list or index out of bounds, use default
                result[topic_id] = 0.5
                continue
            
            result[topic_id] = float(all_probs[topic_idx])
        
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
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'topic_ids': topic_ids,
            'hyperparameters': hyperparameters
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
    
    @classmethod
    def load(cls, model_path: str) -> Tuple['DKTModel', List[int], Dict[str, Any]]:
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
            hidden_size=metadata['hidden_size'],
            num_layers=metadata['num_layers']
        )
        
        # Load state dict
        model.load_state_dict(torch.load(model_path))
        
        return model, metadata['topic_ids'], metadata['hyperparameters']


class DKTTrainer:
    """
    Trainer for the DKT model.
    """
    
    def __init__(
        self,
        model: DKTModel,
        topic_ids: List[int],
        device: str = 'cpu',
        learning_rate: float = 0.001
    ):
        """
        Initialize the trainer.
        
        Args:
            model: DKT model
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
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(input_ids, target_ids, labels)
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
                
                # Forward pass
                outputs = self.model(input_ids, target_ids, labels)
                loss = outputs['loss']
                
                # Get predictions
                logits = outputs['logits']
                batch_size, seq_len, _ = logits.shape
                
                # For each position, get the logit for the target topic
                target_logits = torch.gather(
                    logits, 
                    dim=2, 
                    index=target_ids.unsqueeze(2)
                ).squeeze(2)
                
                # Convert to probabilities
                probs = torch.sigmoid(target_logits)
                
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
