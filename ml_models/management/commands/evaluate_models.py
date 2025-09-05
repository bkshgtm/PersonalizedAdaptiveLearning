from django.core.management.base import BaseCommand
import torch
import numpy as np
import logging
import os
import json
from datetime import datetime
from typing import Dict, Any, List

from ml_models.ml.dkt import DKTModel, DKTTrainer
from ml_models.ml.sakt import SAKTModel, SAKTTrainer
from ml_models.ml.django_data_preparation import DjangoDataPreparation
from ml_models.ml.evaluation_metrics import ComprehensiveEvaluator, BaselineModels

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = 'Comprehensive evaluation of trained knowledge tracing models'

    def add_arguments(self, parser):
        parser.add_argument(
            '--models',
            nargs='+',
            choices=['dkt', 'sakt', 'all'],
            default=['all'],
            help='Which models to evaluate'
        )
        parser.add_argument(
            '--batch-size',
            type=int,
            default=32,
            help='Batch size for evaluation'
        )
        parser.add_argument(
            '--device',
            type=str,
            default='cpu',
            help='Device to use for evaluation (cpu or cuda)'
        )
        parser.add_argument(
            '--include-baselines',
            action='store_true',
            help='Include baseline model comparisons'
        )
        parser.add_argument(
            '--save-predictions',
            action='store_true',
            help='Save model predictions for analysis'
        )
        parser.add_argument(
            '--output-dir',
            type=str,
            default='evaluation_results',
            help='Directory to save evaluation results'
        )

    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS('🔍 Starting Comprehensive Model Evaluation...'))
        
        # Set random seeds for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Initialize data preparation
        data_prep = DjangoDataPreparation()
        num_topics = data_prep.get_num_topics()
        topic_list = data_prep.get_topic_list()
        
        self.stdout.write(f'📊 Found {num_topics} topics: {topic_list[:5]}...')
        
        # Create output directory
        os.makedirs(options['output_dir'], exist_ok=True)
        
        # Prepare data
        self.stdout.write('📋 Preparing evaluation data...')
        train_loader, test_loader = data_prep.create_dataloaders(
            batch_size=options['batch_size']
        )
        
        if len(test_loader) == 0:
            self.stdout.write(
                self.style.ERROR('❌ No test data found. Please ensure you have student interactions in the database.')
            )
            return
        
        models_to_evaluate = options['models']
        if 'all' in models_to_evaluate:
            models_to_evaluate = ['dkt', 'sakt']
        
        evaluation_results = {}
        
        # Evaluate DKT model
        if 'dkt' in models_to_evaluate:
            self.stdout.write(self.style.SUCCESS('🧠 Evaluating DKT Model...'))
            dkt_results = self._evaluate_dkt_model(
                num_topics, train_loader, test_loader, options
            )
            if dkt_results:
                evaluation_results['dkt'] = dkt_results
                self.stdout.write(self.style.SUCCESS('✅ DKT model evaluation completed'))
            else:
                self.stdout.write(self.style.WARNING('⚠️ DKT model evaluation failed'))
        
        # Evaluate SAKT model
        if 'sakt' in models_to_evaluate:
            self.stdout.write(self.style.SUCCESS('🎯 Evaluating SAKT Model...'))
            sakt_results = self._evaluate_sakt_model(
                num_topics, train_loader, test_loader, options
            )
            if sakt_results:
                evaluation_results['sakt'] = sakt_results
                self.stdout.write(self.style.SUCCESS('✅ SAKT model evaluation completed'))
            else:
                self.stdout.write(self.style.WARNING('⚠️ SAKT model evaluation failed'))
        
        # Generate comparison report if multiple models evaluated
        if len(evaluation_results) > 1:
            self.stdout.write(self.style.SUCCESS('📊 Generating model comparison report...'))
            self._generate_comparison_report(evaluation_results, options)
        
        # Save comprehensive evaluation summary
        self._save_evaluation_summary(evaluation_results, options)
        
        self.stdout.write(self.style.SUCCESS('🎉 Comprehensive evaluation completed!'))
        self.stdout.write('')
        self.stdout.write(f'📁 Results saved to: {options["output_dir"]}/')
        self.stdout.write('📈 Check the evaluation reports for detailed analysis')

    def _evaluate_dkt_model(self, num_topics: int, train_loader, test_loader, options: Dict[str, Any]):
        """Evaluate the DKT model."""
        model_path = 'trained_models/dkt_model.pth'
        
        if not os.path.exists(model_path):
            self.stdout.write(f'   ❌ DKT model not found at {model_path}')
            return None
        
        try:
            # Load model
            dkt_model, topic_ids, hyperparameters = DKTModel.load(model_path)
            
            # Initialize trainer with comprehensive evaluation
            trainer = DKTTrainer(
                model=dkt_model,
                topic_ids=topic_ids,
                device=options['device'],
                learning_rate=0.001,  # Not used for evaluation
                enable_comprehensive_eval=True
            )
            
            # Run comprehensive evaluation
            results = trainer.evaluate(
                test_loader=test_loader,
                train_loader=train_loader if options['include_baselines'] else None,
                include_baselines=options['include_baselines']
            )
            
            # Save predictions if requested
            if options['save_predictions']:
                self._save_model_predictions(
                    trainer, test_loader, 'dkt', options['output_dir']
                )
            
            # Log key metrics
            if 'classification_metrics' in results:
                cm = results['classification_metrics']
                self.stdout.write(f'   📈 Accuracy: {cm["accuracy"]:.4f}')
                self.stdout.write(f'   📈 AUC: {cm["auc"]:.4f}')
                self.stdout.write(f'   📈 F1-Score: {cm["f1"]:.4f}')
            
            return results
            
        except Exception as e:
            self.stdout.write(f'   ❌ Error evaluating DKT model: {e}')
            logger.error(f"DKT evaluation error: {e}")
            return None

    def _evaluate_sakt_model(self, num_topics: int, train_loader, test_loader, options: Dict[str, Any]):
        """Evaluate the SAKT model."""
        model_path = 'trained_models/sakt_model.pth'
        
        if not os.path.exists(model_path):
            self.stdout.write(f'   ❌ SAKT model not found at {model_path}')
            return None
        
        try:
            # Load model
            sakt_model, topic_ids, hyperparameters = SAKTModel.load(model_path)
            
            # Initialize trainer with comprehensive evaluation
            trainer = SAKTTrainer(
                model=sakt_model,
                topic_ids=topic_ids,
                device=options['device'],
                learning_rate=0.001,  # Not used for evaluation
                enable_comprehensive_eval=True
            )
            
            # Run comprehensive evaluation
            results = trainer.evaluate(
                test_loader=test_loader,
                train_loader=train_loader if options['include_baselines'] else None,
                include_baselines=options['include_baselines']
            )
            
            # Save predictions if requested
            if options['save_predictions']:
                self._save_model_predictions(
                    trainer, test_loader, 'sakt', options['output_dir']
                )
            
            # Log key metrics
            if 'classification_metrics' in results:
                cm = results['classification_metrics']
                self.stdout.write(f'   📈 Accuracy: {cm["accuracy"]:.4f}')
                self.stdout.write(f'   📈 AUC: {cm["auc"]:.4f}')
                self.stdout.write(f'   📈 F1-Score: {cm["f1"]:.4f}')
            
            return results
            
        except Exception as e:
            self.stdout.write(f'   ❌ Error evaluating SAKT model: {e}')
            logger.error(f"SAKT evaluation error: {e}")
            return None

    def _save_model_predictions(self, trainer, test_loader, model_name: str, output_dir: str):
        """Save model predictions for detailed analysis."""
        predictions = []
        
        trainer.model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                # Get batch data
                if model_name == 'dkt':
                    input_ids = batch['input_ids'].to(trainer.device)
                    target_ids = batch['target_ids'].to(trainer.device)
                    labels = batch['labels'].to(trainer.device)
                    
                    outputs = trainer.model(input_ids, target_ids, labels)
                    probs = torch.sigmoid(outputs['logits'])
                    
                elif model_name == 'sakt':
                    input_ids = batch['input_ids'].to(trainer.device)
                    target_ids = batch['target_ids'].to(trainer.device)
                    labels = batch['labels'].to(trainer.device)
                    attention_mask = batch['attention_mask'].to(trainer.device)
                    
                    # Create input labels
                    input_labels = torch.zeros_like(labels)
                    input_labels[:, 1:] = labels[:, :-1]
                    labels = labels * attention_mask
                    
                    outputs = trainer.model(input_ids, input_labels, target_ids, labels)
                    probs = torch.sigmoid(outputs['logits'])
                
                # Store predictions
                batch_predictions = {
                    'batch_idx': batch_idx,
                    'predictions': probs.cpu().numpy().tolist(),
                    'labels': labels.cpu().numpy().tolist(),
                    'topic_ids': target_ids.cpu().numpy().tolist()
                }
                
                if 'student_ids' in batch:
                    batch_predictions['student_ids'] = batch['student_ids'].cpu().numpy().tolist()
                
                predictions.append(batch_predictions)
        
        # Save predictions
        predictions_file = os.path.join(output_dir, f'{model_name}_predictions.json')
        with open(predictions_file, 'w') as f:
            json.dump(predictions, f, indent=2)
        
        self.stdout.write(f'   💾 Predictions saved to {predictions_file}')

    def _generate_comparison_report(self, evaluation_results: Dict[str, Any], options: Dict[str, Any]):
        """Generate a comparison report between models."""
        
        # Extract key metrics for comparison
        comparison_data = {}
        for model_name, results in evaluation_results.items():
            if 'classification_metrics' in results:
                cm = results['classification_metrics']
                cal = results.get('calibration_metrics', {})
                
                comparison_data[model_name] = {
                    'accuracy': cm.get('accuracy', 0.0),
                    'precision': cm.get('precision', 0.0),
                    'recall': cm.get('recall', 0.0),
                    'f1': cm.get('f1', 0.0),
                    'auc': cm.get('auc', 0.0),
                    'brier_score': cal.get('brier_score', float('inf')),
                    'expected_calibration_error': cal.get('expected_calibration_error', float('inf'))
                }
        
        # Generate comparison report
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("MODEL COMPARISON REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Models compared: {', '.join(comparison_data.keys())}")
        report_lines.append("")
        
        # Metrics comparison table
        report_lines.append("PERFORMANCE METRICS COMPARISON")
        report_lines.append("-" * 50)
        
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc', 'brier_score', 'expected_calibration_error']
        
        # Header
        header = f"{'Metric':<25}"
        for model_name in comparison_data.keys():
            header += f"{model_name.upper():<12}"
        header += "Best"
        report_lines.append(header)
        report_lines.append("-" * len(header))
        
        # Metrics rows
        for metric in metrics:
            row = f"{metric.replace('_', ' ').title():<25}"
            values = []
            
            for model_name in comparison_data.keys():
                value = comparison_data[model_name][metric]
                if value == float('inf'):
                    row += f"{'N/A':<12}"
                    values.append(None)
                else:
                    row += f"{value:<12.4f}"
                    values.append(value)
            
            # Determine best model for this metric
            valid_values = [(i, v) for i, v in enumerate(values) if v is not None]
            if valid_values:
                if metric in ['brier_score', 'expected_calibration_error']:
                    # Lower is better
                    best_idx = min(valid_values, key=lambda x: x[1])[0]
                else:
                    # Higher is better
                    best_idx = max(valid_values, key=lambda x: x[1])[0]
                
                best_model = list(comparison_data.keys())[best_idx]
                row += f"{best_model.upper()}"
            else:
                row += "N/A"
            
            report_lines.append(row)
        
        report_lines.append("")
        
        # Summary and recommendations
        report_lines.append("SUMMARY AND RECOMMENDATIONS")
        report_lines.append("-" * 40)
        
        # Count wins for each model
        model_wins = {model: 0 for model in comparison_data.keys()}
        for metric in metrics:
            values = []
            for model_name in comparison_data.keys():
                value = comparison_data[model_name][metric]
                if value != float('inf'):
                    values.append((model_name, value))
            
            if values:
                if metric in ['brier_score', 'expected_calibration_error']:
                    best_model = min(values, key=lambda x: x[1])[0]
                else:
                    best_model = max(values, key=lambda x: x[1])[0]
                model_wins[best_model] += 1
        
        # Overall best model
        overall_best = max(model_wins.items(), key=lambda x: x[1])
        report_lines.append(f"Overall best model: {overall_best[0].upper()} (wins in {overall_best[1]}/{len(metrics)} metrics)")
        
        # Specific recommendations
        report_lines.append("")
        report_lines.append("Specific observations:")
        
        for model_name, data in comparison_data.items():
            strengths = []
            weaknesses = []
            
            if data['accuracy'] > 0.5:
                strengths.append("good accuracy")
            if data['auc'] > 0.7:
                strengths.append("strong AUC")
            if data['f1'] > 0.5:
                strengths.append("balanced precision/recall")
            if data['brier_score'] < 0.25:
                strengths.append("well-calibrated")
            
            if data['accuracy'] < 0.5:
                weaknesses.append("low accuracy")
            if data['auc'] < 0.6:
                weaknesses.append("poor discrimination")
            if data['expected_calibration_error'] > 0.1:
                weaknesses.append("poorly calibrated")
            
            report_lines.append(f"• {model_name.upper()}:")
            if strengths:
                report_lines.append(f"  Strengths: {', '.join(strengths)}")
            if weaknesses:
                report_lines.append(f"  Areas for improvement: {', '.join(weaknesses)}")
        
        report_lines.append("")
        report_lines.append("=" * 80)
        
        # Save comparison report
        report_file = os.path.join(options['output_dir'], 'model_comparison_report.txt')
        with open(report_file, 'w') as f:
            f.write('\n'.join(report_lines))
        
        self.stdout.write(f'   📊 Comparison report saved to {report_file}')

    def _save_evaluation_summary(self, evaluation_results: Dict[str, Any], options: Dict[str, Any]):
        """Save a comprehensive evaluation summary."""
        
        summary = {
            'evaluation_timestamp': datetime.now().isoformat(),
            'evaluation_config': {
                'models_evaluated': list(evaluation_results.keys()),
                'include_baselines': options['include_baselines'],
                'batch_size': options['batch_size'],
                'device': options['device']
            },
            'results': evaluation_results
        }
        
        # Save summary
        summary_file = os.path.join(options['output_dir'], 'evaluation_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        self.stdout.write(f'   📋 Evaluation summary saved to {summary_file}')
