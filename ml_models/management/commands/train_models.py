from django.core.management.base import BaseCommand
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import logging
import os
from typing import Dict, Any

from ml_models.ml.dkt import DKTModel, DKTTrainer
from ml_models.ml.sakt import SAKTModel, SAKTTrainer
from ml_models.ml.django_data_preparation import DjangoDataPreparation
from learning_paths.ml.adaptive_path_lstm import AdaptiveLearningPathLSTM

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = 'Train DKT, SAKT, and Enhanced LSTM models'

    def add_arguments(self, parser):
        parser.add_argument(
            '--epochs',
            type=int,
            default=20,
            help='Number of training epochs'
        )
        parser.add_argument(
            '--batch-size',
            type=int,
            default=32,
            help='Batch size for training'
        )
        parser.add_argument(
            '--learning-rate',
            type=float,
            default=0.001,
            help='Learning rate for training'
        )
        parser.add_argument(
            '--device',
            type=str,
            default='cpu',
            help='Device to use for training (cpu or cuda)'
        )
        parser.add_argument(
            '--models',
            nargs='+',
            choices=['dkt', 'sakt', 'lstm', 'all'],
            default=['all'],
            help='Which models to train'
        )

    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS('🚀 Starting Model Training...'))
        
        # Initialize data preparation
        data_prep = DjangoDataPreparation()
        num_topics = data_prep.get_num_topics()
        topic_list = data_prep.get_topic_list()
        
        self.stdout.write(f'📊 Found {num_topics} topics: {topic_list[:5]}...')
        
        # Create models directory if it doesn't exist
        os.makedirs('trained_models', exist_ok=True)
        
        # Prepare data for DKT and SAKT training
        self.stdout.write('📋 Preparing training data...')
        train_loader, test_loader = data_prep.create_dataloaders(
            batch_size=options['batch_size']
        )
        
        if len(train_loader) == 0:
            self.stdout.write(
                self.style.ERROR('❌ No training data found. Please ensure you have student interactions in the database.')
            )
            return
        
        models_to_train = options['models']
        if 'all' in models_to_train:
            models_to_train = ['dkt', 'sakt', 'lstm']
        
        # Train DKT model
        if 'dkt' in models_to_train:
            self.stdout.write(self.style.SUCCESS(' Training DKT Model...'))
            dkt_model = self._train_dkt_model(
                num_topics, train_loader, test_loader, options
            )
            self.stdout.write(self.style.SUCCESS(' DKT model training completed'))
        
        # Train SAKT model
        if 'sakt' in models_to_train:
            self.stdout.write(self.style.SUCCESS(' Training SAKT Model...'))
            sakt_model = self._train_sakt_model(
                num_topics, train_loader, test_loader, options
            )
            self.stdout.write(self.style.SUCCESS(' SAKT model training completed'))
        
        # Train Enhanced LSTM model
        if 'lstm' in models_to_train:
            self.stdout.write(self.style.SUCCESS(' Training Enhanced LSTM Model...'))
            lstm_model = self._train_lstm_model(
                num_topics, data_prep, options
            )
            self.stdout.write(self.style.SUCCESS('Enhanced LSTM model training completed'))
        
        self.stdout.write(self.style.SUCCESS('🎉 All model training completed!'))
        self.stdout.write('')
        self.stdout.write('📁 Trained models saved to:')
        if 'dkt' in models_to_train:
            self.stdout.write('   • trained_models/dkt_model.pth')
        if 'sakt' in models_to_train:
            self.stdout.write('   • trained_models/sakt_model.pth')
        if 'lstm' in models_to_train:
            self.stdout.write('   • trained_models/lstm_model.pth')
        self.stdout.write('')
        self.stdout.write('🎯 Next step: Generate learning paths with:')
        self.stdout.write(self.style.WARNING('   python manage.py generate_learning_paths --student-id A00000001'))

    def _train_dkt_model(self, num_topics: int, train_loader: DataLoader, 
                        test_loader: DataLoader, options: Dict[str, Any]) -> DKTModel:
        """Train the DKT model."""
        # Initialize model
        dkt_model = DKTModel(
            num_topics=num_topics,
            hidden_size=128,
            num_layers=2,
            dropout=0.2
        )
        
        # Initialize trainer
        trainer = DKTTrainer(
            model=dkt_model,
            topic_ids=list(range(1, num_topics + 1)),
            device=options['device'],
            learning_rate=options['learning_rate'],
            enable_comprehensive_eval=True
        )
        
        # Train model with comprehensive evaluation
        history = trainer.train(
            train_loader=train_loader,
            test_loader=test_loader,
            num_epochs=options['epochs'],
            final_comprehensive_eval=True,
            include_baselines=True
        )
        
        # Save model
        dkt_model.save(
            model_path='trained_models/dkt_model.pth',
            topic_ids=list(range(1, num_topics + 1)),
            hyperparameters={
                'num_topics': num_topics,
                'hidden_size': 128,
                'num_layers': 2,
                'dropout': 0.2,
                'learning_rate': options['learning_rate'],
                'epochs': options['epochs']
            }
        )
        
        self.stdout.write(f'  Final test accuracy: {history["test_accuracy"][-1]:.4f}')
        return dkt_model

    def _train_sakt_model(self, num_topics: int, train_loader: DataLoader, 
                         test_loader: DataLoader, options: Dict[str, Any]) -> SAKTModel:
        """Train the SAKT model."""
        # Initialize model
        sakt_model = SAKTModel(
            num_topics=num_topics,
            d_model=64,
            n_heads=4,
            dropout=0.2
        )
        
        # Initialize trainer
        trainer = SAKTTrainer(
            model=sakt_model,
            topic_ids=list(range(1, num_topics + 1)),
            device=options['device'],
            learning_rate=options['learning_rate'],
            enable_comprehensive_eval=True
        )
        
        # Train model with comprehensive evaluation
        history = trainer.train(
            train_loader=train_loader,
            test_loader=test_loader,
            num_epochs=options['epochs'],
            final_comprehensive_eval=True,
            include_baselines=True
        )
        
        # Save model
        sakt_model.save(
            model_path='trained_models/sakt_model.pth',
            topic_ids=list(range(1, num_topics + 1)),
            hyperparameters={
                'num_topics': num_topics,
                'd_model': 64,
                'n_heads': 4,
                'dropout': 0.2,
                'learning_rate': options['learning_rate'],
                'epochs': options['epochs']
            }
        )
        
        self.stdout.write(f'   Final test accuracy: {history["test_accuracy"][-1]:.4f}')
        return sakt_model

    def _train_lstm_model(self, num_topics: int, data_prep: DjangoDataPreparation, 
                         options: Dict[str, Any]) -> AdaptiveLearningPathLSTM:
        """Train the Enhanced LSTM model."""
        from core.models import Student
        import torch.optim as optim
        
        # Initialize model
        lstm_model = AdaptiveLearningPathLSTM(
            num_topics=num_topics,
            hidden_size=128,
            num_layers=2,
            dropout=0.3
        )
        
        # Set up training
        device = torch.device(options['device'])
        lstm_model.to(device)
        optimizer = optim.Adam(lstm_model.parameters(), lr=options['learning_rate'])
        criterion = nn.CrossEntropyLoss()
        
        # Get all students for training
        students = Student.objects.filter(interactions__isnull=False).distinct()
        
        if students.count() == 0:
            self.stdout.write('   ⚠️ No students with interactions found for LSTM training')
            # Save initialized model
            model_path = 'trained_models/adaptive_path_lstm.pth'
            torch.save({
                'model_state_dict': lstm_model.state_dict(),
                'num_topics': num_topics,
                'hidden_size': 128,
                'num_layers': 2,
                'dropout': 0.3,
                'hyperparameters': {
                    'num_topics': num_topics,
                    'hidden_size': 128,
                    'num_layers': 2,
                    'dropout': 0.3,
                    'learning_rate': options['learning_rate'],
                    'epochs': options['epochs']
                }
            }, model_path)
            return lstm_model
        
        self.stdout.write(f'   Training LSTM with {students.count()} students...')
        
        # Training loop
        lstm_model.train()
        for epoch in range(options['epochs']):
            total_loss = 0
            num_batches = 0
            
            for student in students:
                try:
                    # Get student features and mastery scores
                    student_features = self._get_student_features_for_lstm(student, data_prep)
                    mastery_scores = self._get_mastery_scores_for_lstm(student, data_prep, num_topics)
                    
                    if student_features is None or mastery_scores is None:
                        continue
                    
                    # Create training sequence (simplified for demonstration)
                    # In practice, you'd use actual learning sequences
                    seq_len = 5
                    topic_sequence = torch.zeros(1, seq_len, num_topics)
                    
                    # Add some random topic selections for training
                    for i in range(seq_len):
                        # Select topics with lower mastery scores as targets
                        weak_topics = (mastery_scores < 0.6).nonzero(as_tuple=True)[0]
                        if len(weak_topics) > 0:
                            target_topic = weak_topics[torch.randint(0, len(weak_topics), (1,))]
                            topic_sequence[0, i, target_topic] = 1.0
                    
                    # Move to device
                    student_features = student_features.to(device)
                    mastery_scores = mastery_scores.to(device)
                    topic_sequence = topic_sequence.to(device)
                    
                    # Forward pass
                    outputs = lstm_model(student_features.unsqueeze(0), mastery_scores.unsqueeze(0), topic_sequence)
                    
                    # Create target (next topic in sequence)
                    target = torch.zeros(1, seq_len, num_topics).to(device)
                    for i in range(seq_len - 1):
                        target[0, i] = topic_sequence[0, i + 1]
                    
                    # Calculate loss
                    loss = criterion(
                        outputs['next_topic_probs'].view(-1, num_topics),
                        target.view(-1, num_topics).argmax(dim=-1)
                    )
                    
                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    num_batches += 1
                    
                except Exception as e:
                    # Skip problematic students
                    continue
            
            if num_batches > 0:
                avg_loss = total_loss / num_batches
                if epoch % 5 == 0:
                    self.stdout.write(f'   📈 Epoch {epoch}/{options["epochs"]}, Loss: {avg_loss:.4f}')
        
        # Save trained model
        model_path = 'trained_models/adaptive_path_lstm.pth'
        torch.save({
            'model_state_dict': lstm_model.state_dict(),
            'num_topics': num_topics,
            'hidden_size': 128,
            'num_layers': 2,
            'dropout': 0.3,
            'hyperparameters': {
                'num_topics': num_topics,
                'hidden_size': 128,
                'num_layers': 2,
                'dropout': 0.3,
                'learning_rate': options['learning_rate'],
                'epochs': options['epochs']
            }
        }, model_path)
        
        self.stdout.write(f'    LSTM model training completed and saved')
        return lstm_model
    
    def _get_student_features_for_lstm(self, student, data_prep):
        """Get student features for LSTM training."""
        try:
            # Map categorical fields to numerical values
            academic_level_map = {'freshman': 1, 'sophomore': 2, 'junior': 3, 'senior': 4, 'graduate': 5}
            study_freq_map = {'rarely': 1, 'monthly': 2, 'biweekly': 3, 'weekly': 4, 'daily': 5}
            
            features = [
                academic_level_map.get(student.academic_level, 3),
                student.gpa / 4.0,
                student.prior_knowledge_score or 0.5,
                study_freq_map.get(student.study_frequency, 3),
                student.attendance_rate / 100.0,
                student.participation_score / 100.0,
                min(student.total_time_spent.total_seconds() / 3600, 100) / 100 if student.total_time_spent else 0.1,
                min(student.average_time_per_session.total_seconds() / 3600, 5) / 5 if student.average_time_per_session else 0.2,
                1.0,
                0.5
            ]
            
            return torch.tensor(features, dtype=torch.float32)
        except Exception:
            return None
    
    def _get_mastery_scores_for_lstm(self, student, data_prep, num_topics):
        """Get mastery scores for LSTM training."""
        try:
            mastery_scores = torch.zeros(num_topics)
            
            # Get student state
            student_state = data_prep.get_student_current_state(student.student_id)
            if student_state and 'mastery_scores' in student_state:
                for topic_name, mastery_score in student_state['mastery_scores'].items():
                    if topic_name in data_prep.topic_to_id:
                        topic_idx = data_prep.topic_to_id[topic_name] - 1
                        if 0 <= topic_idx < num_topics:
                            mastery_scores[topic_idx] = mastery_score
            
            return mastery_scores
        except Exception:
            return None
