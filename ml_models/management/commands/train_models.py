import logging
from django.core.management.base import BaseCommand
from ml_models.ml.data_preparation import prepare_training_data
from ml_models.ml.dkt import DKTModel
from ml_models.ml.sakt import SAKTModel
from ml_models.ml.metrics import log_metrics
from django.conf import settings
import os
import time

logger = logging.getLogger(__name__)

class Command(BaseCommand):
    help = 'Train knowledge tracing models and save results'

    def add_arguments(self, parser):
        parser.add_argument('--model', type=str, default='dkt',
                          help='Model to train (dkt or sakt)')
        parser.add_argument('--epochs', type=int, default=10,
                          help='Number of training epochs')
        parser.add_argument('--batch_size', type=int, default=32,
                          help='Training batch size')
        parser.add_argument('--log_dir', type=str, 
                          default=os.path.join(settings.BASE_DIR, 'training_logs'),
                          help='Directory to save training logs')

    def handle(self, *args, **options):
        model_type = options['model']
        epochs = options['epochs']
        batch_size = options['batch_size']
        log_dir = options['log_dir']

        # Ensure log directory exists
        os.makedirs(log_dir, exist_ok=True)

        # Prepare data
        self.stdout.write("Preparing training data...")
        train_data, val_data = prepare_training_data()

        # Initialize model
        if model_type == 'dkt':
            model = DKTModel()
        elif model_type == 'sakt':
            model = SAKTModel()
        else:
            self.stderr.write(f"Unknown model type: {model_type}")
            return

        # Train model
        self.stdout.write(f"Training {model_type.upper()} model for {epochs} epochs...")
        start_time = time.time()
        
        history = model.train(
            train_data,
            validation_data=val_data,
            epochs=epochs,
            batch_size=batch_size
        )

        # Log results
        training_time = time.time() - start_time
        log_file = os.path.join(log_dir, f"{model_type}_training_{int(start_time)}.log")
        
        with open(log_file, 'w') as f:
            f.write(f"Training completed in {training_time:.2f} seconds\n")
            f.write(f"Final metrics:\n")
            for metric, value in history.history.items():
                f.write(f"{metric}: {value[-1]:.4f}\n")

        self.stdout.write(self.style.SUCCESS(
            f"Successfully trained {model_type.upper()} model\n"
            f"Training log saved to {log_file}"
        ))
