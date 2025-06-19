from django.core.management.base import BaseCommand
from django.db import transaction
import torch
import json
import logging
from typing import Dict, List, Any, Optional

from core.models import Student, Topic, KnowledgeState
from ml_models.ml.dkt import DKTModel
from ml_models.ml.sakt import SAKTModel
from ml_models.ml.django_data_preparation import DjangoDataPreparation
from learning_paths.ml.adaptive_path_lstm import AdaptiveLearningPathLSTM, DjangoIntegratedPathGenerator

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = 'Generate personalized learning paths using trained ML models'

    def add_arguments(self, parser):
        parser.add_argument(
            '--student-id',
            type=str,
            help='Generate learning path for specific student'
        )
        parser.add_argument(
            '--all-students',
            action='store_true',
            help='Generate learning paths for all students'
        )
        parser.add_argument(
            '--update-predictions',
            action='store_true',
            help='Update mastery predictions before generating paths'
        )
        parser.add_argument(
            '--no-database',
            action='store_true',
            help='Skip saving learning paths to database (default is to save)'
        )
        parser.add_argument(
            '--output-format',
            choices=['json', 'summary', 'detailed'],
            default='detailed',
            help='Output format for learning paths'
        )

    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS('🎯 Starting Learning Path Generation...'))
        
        # Validate arguments
        if not options['student_id'] and not options['all_students']:
            self.stdout.write(
                self.style.ERROR('❌ Please specify either --student-id or --all-students')
            )
            return
        
        # Initialize data preparation
        data_prep = DjangoDataPreparation()
        
        # Update predictions if requested
        if options['update_predictions']:
            self.stdout.write('🔄 Updating mastery predictions...')
            self._update_mastery_predictions(data_prep)
        
        # Generate learning paths
        if options['student_id']:
            self._generate_single_student_path(options['student_id'], options)
        elif options['all_students']:
            self._generate_all_student_paths(options)
        
        self.stdout.write(self.style.SUCCESS('🎉 Learning path generation completed!'))

    def _update_mastery_predictions(self, data_prep: DjangoDataPreparation):
        """Update mastery predictions using trained models."""
        try:
            # Load trained models
            self.stdout.write('📂 Loading trained models...')
            dkt_model, topic_ids, _ = DKTModel.load('trained_models/dkt_model.pth')
            dkt_model.eval()
            
            sakt_model, _, _ = SAKTModel.load('trained_models/sakt_model.pth')
            sakt_model.eval()
            
            self.stdout.write('✅ Models loaded successfully')
            
        except Exception as e:
            self.stdout.write(self.style.WARNING(f'⚠️ Could not load trained models: {e}'))
            self.stdout.write('   Using existing predictions from database')
            return
        
        # Get all students with interactions
        students = Student.objects.filter(interactions__isnull=False).distinct()
        
        self.stdout.write(f'🔄 Updating predictions for {students.count()} students...')
        
        for student in students:
            try:
                # Get student's current state
                student_state = data_prep.get_student_current_state(student.student_id)
                
                if not student_state or len(student_state['interaction_sequence']) < 3:
                    continue
                
                # Prepare data for models
                interaction_seq = student_state['interaction_sequence']
                input_ids = torch.tensor([int(item['topic_id']) for item in interaction_seq])
                input_labels = torch.tensor([1 if item['correct'] else 0 for item in interaction_seq])
                
                # Get DKT predictions
                dkt_predictions = dkt_model.predict(
                    input_ids=input_ids,
                    input_labels=input_labels,
                    topic_ids=topic_ids
                )
                
                # Get SAKT predictions
                sakt_predictions = sakt_model.predict(
                    input_ids=input_ids,
                    input_labels=input_labels,
                    topic_ids=topic_ids
                )
                
                # Combine predictions (ensemble)
                combined_predictions = {}
                for topic_id in topic_ids:
                    if topic_id in dkt_predictions and topic_id in sakt_predictions:
                        # Average the predictions
                        combined_score = (dkt_predictions[topic_id] + sakt_predictions[topic_id]) / 2
                        
                        # Map topic_id back to topic name
                        topic_list = data_prep.get_topic_list()
                        if topic_id <= len(topic_list):
                            topic_name = topic_list[topic_id - 1]
                            combined_predictions[topic_name] = combined_score
                
                # Update knowledge states in database
                data_prep.update_knowledge_states(student.student_id, combined_predictions)
                
                self.stdout.write(f'   ✅ Updated predictions for {student.student_id}')
                
            except Exception as e:
                self.stdout.write(
                    self.style.WARNING(f'   ⚠️ Error processing {student.student_id}: {e}')
                )

    def _generate_single_student_path(self, student_id: str, options: Dict[str, Any]):
        """Generate learning path for a single student."""
        try:
            student = Student.objects.get(student_id=student_id)
        except Student.DoesNotExist:
            self.stdout.write(self.style.ERROR(f'❌ Student {student_id} not found'))
            return
        
        self.stdout.write(f'🎯 Generating learning path for student {student_id}...')
        
        # Initialize path generator
        try:
            path_generator = DjangoIntegratedPathGenerator()
            
            # Generate learning path
            learning_path = path_generator.generate_comprehensive_learning_path(student_id)
            
            if learning_path:
                self._display_learning_path(learning_path, options['output_format'])
                
                # Save to database by default (unless --no-database flag is used)
                if not options['no_database']:
                    self._save_learning_path_to_database(learning_path)
                    
            else:
                self.stdout.write(self.style.WARNING('⚠️ No learning path generated'))
                
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'❌ Error generating path: {e}'))

    def _generate_all_student_paths(self, options: Dict[str, Any]):
        """Generate learning paths for all students."""
        students = Student.objects.filter(interactions__isnull=False).distinct()
        
        if not students:
            self.stdout.write(self.style.WARNING('⚠️ No students with interactions found'))
            return
        
        self.stdout.write(f'🎯 Generating learning paths for {students.count()} students...')
        
        path_generator = DjangoIntegratedPathGenerator()
        
        for student in students:
            try:
                self.stdout.write(f'\n📚 Student: {student.student_id}')
                
                learning_path = path_generator.generate_comprehensive_learning_path(
                    student.student_id
                )
                
                if learning_path:
                    if options['output_format'] == 'summary':
                        self._display_path_summary(learning_path)
                    else:
                        self._display_learning_path(learning_path, options['output_format'])
                        
                    # Save to database by default (unless --no-database flag is used)
                    if not options['no_database']:
                        self._save_learning_path_to_database(learning_path)
                else:
                    self.stdout.write('   ⚠️ No path generated')
                    
            except Exception as e:
                self.stdout.write(f'   ❌ Error: {e}')

    def _display_learning_path(self, learning_path: Dict[str, Any], output_format: str):
        """Display the learning path in the specified format."""
        if output_format == 'json':
            self.stdout.write(json.dumps(learning_path, indent=2, default=str))
            return
        
        # Display detailed or summary format
        stats = learning_path['student_stats']
        self.stdout.write(f"👤 Student Profile:")
        self.stdout.write(f"   • Academic Level: {stats['academic_level']}")
        self.stdout.write(f"   • GPA: {stats['gpa']}")
        self.stdout.write(f"   • Major: {stats['major']}")
        self.stdout.write(f"   • Total Interactions: {stats['total_interactions']}")
        self.stdout.write(f"   • Average Performance: {stats['average_performance']:.2f}")
        
        # Weak Topics
        weak_topics = learning_path['weak_topics']
        self.stdout.write(f"\n📉 Weak Topics ({len(weak_topics)}):")
        for topic in weak_topics[:5]:  # Show first 5
            self.stdout.write(f"   • {topic['name']} (mastery: {topic['current_mastery']:.2f})")
            if topic['prerequisites']:
                self.stdout.write(f"     Prerequisites: {', '.join(topic['prerequisites'])}")
            self.stdout.write(f"     Resources ({len(topic['resources'])} available):")
            for i, resource in enumerate(topic['resources'][:3], 1):  # Show first 3 resources
                self.stdout.write(f"       {i}. {resource['title']} ({resource['type']}, {resource['difficulty']})")
                self.stdout.write(f"          URL: {resource['url']}")
                self.stdout.write(f"          Time: {resource['estimated_time']:.1f}h - {resource['description'][:80]}...")
        
        # Recommended Path
        recommendations = learning_path['recommended_path']
        self.stdout.write(f"\n🎯 Recommended Learning Path ({len(recommendations)} topics):")
        for i, rec in enumerate(recommendations[:5], 1):  # Show first 5
            status = "⚠️ Prerequisites needed" if rec['should_study_prerequisites_first'] else "✅ Ready to study"
            self.stdout.write(f"   {i}. {rec['topic']} (confidence: {rec['confidence']:.3f})")
            self.stdout.write(f"      {status}")
            self.stdout.write(f"      Difficulty: {rec['recommended_difficulty']}")
            self.stdout.write(f"      Est. Time: {rec['estimated_time_hours']:.1f} hours")
            
            if rec['unmet_prerequisites']:
                self.stdout.write(f"      Missing prerequisites: {', '.join(rec['unmet_prerequisites'])}")
            
            self.stdout.write(f"      Resources ({len(rec['resources'])} available):")
            for j, resource in enumerate(rec['resources'][:3], 1):  # Show first 3 resources
                self.stdout.write(f"        {j}. {resource['title']} ({resource['type']}, {resource['difficulty']})")
                self.stdout.write(f"           URL: {resource['url']}")
                self.stdout.write(f"           Time: {resource['estimated_time']:.1f}h - {resource['description'][:80]}...")
        
        # Summary
        total_time = learning_path['total_estimated_time']
        self.stdout.write(f"\n⏱️  Total Estimated Learning Time: {total_time:.1f} hours")

    def _display_path_summary(self, learning_path: Dict[str, Any]):
        """Display a brief summary of the learning path."""
        stats = learning_path['student_stats']
        weak_count = len(learning_path['weak_topics'])
        rec_count = len(learning_path['recommended_path'])
        total_time = learning_path['total_estimated_time']
        
        self.stdout.write(f"   📊 Performance: {stats['average_performance']:.2f}")
        self.stdout.write(f"   📉 Weak Topics: {weak_count}")
        self.stdout.write(f"   🎯 Recommendations: {rec_count}")
        self.stdout.write(f"   ⏱️  Est. Time: {total_time:.1f}h")
        
        if learning_path['recommended_path']:
            top_rec = learning_path['recommended_path'][0]
            self.stdout.write(f"   🔝 Top Priority: {top_rec['topic']}")

    def _save_learning_path_to_database(self, learning_path: Dict[str, Any]):
        """Save learning path to database using the new models."""
        from learning_paths.models import LearningPath, WeakTopic, RecommendedTopic, TopicResource
        from core.models import Student, Course, Topic
        from django.db import transaction
        
        try:
            with transaction.atomic():
                # Get student and course
                student = Student.objects.get(student_id=learning_path['student_id'])
                # Assume CS206 course for now - you can make this dynamic
                course = Course.objects.get(course_id='CS206')
                
                # Create the main learning path
                path = LearningPath.objects.create(
                    student=student,
                    course=course,
                    name=f"Learning Path for {student.student_id}",
                    description="AI-generated personalized learning path",
                    student_stats=learning_path['student_stats'],
                    total_estimated_time=learning_path['total_estimated_time'],
                    weak_topics_count=len(learning_path['weak_topics']),
                    recommended_topics_count=len(learning_path['recommended_path'])
                )
                
                # Save weak topics
                for order, weak_topic_data in enumerate(learning_path['weak_topics']):
                    try:
                        topic = Topic.objects.get(name=weak_topic_data['name'])
                        weak_topic = WeakTopic.objects.create(
                            learning_path=path,
                            topic=topic,
                            current_mastery=weak_topic_data['current_mastery'],
                            prerequisites=weak_topic_data['prerequisites'],
                            related_topics=weak_topic_data['related_topics'],
                            order=order
                        )
                        
                        # Save resources for weak topics
                        for res_order, resource_data in enumerate(weak_topic_data['resources']):
                            TopicResource.objects.create(
                                weak_topic=weak_topic,
                                title=resource_data['title'],
                                description=resource_data['description'],
                                url=resource_data['url'],
                                resource_type=resource_data['type'],
                                difficulty=resource_data['difficulty'],
                                estimated_time=resource_data['estimated_time'],
                                order=res_order
                            )
                    except Topic.DoesNotExist:
                        self.stdout.write(f"   ⚠️ Topic not found: {weak_topic_data['name']}")
                
                # Save recommended topics
                for priority, rec_topic_data in enumerate(learning_path['recommended_path'], 1):
                    try:
                        topic = Topic.objects.get(name=rec_topic_data['topic'])
                        recommended_topic = RecommendedTopic.objects.create(
                            learning_path=path,
                            topic=topic,
                            confidence=rec_topic_data['confidence'],
                            recommended_difficulty=rec_topic_data['recommended_difficulty'],
                            estimated_time_hours=rec_topic_data['estimated_time_hours'],
                            prerequisites=rec_topic_data['prerequisites'],
                            unmet_prerequisites=rec_topic_data['unmet_prerequisites'],
                            should_study_prerequisites_first=rec_topic_data['should_study_prerequisites_first'],
                            related_topics=rec_topic_data['related_topics'],
                            priority=priority
                        )
                        
                        # Save resources for recommended topics
                        for res_order, resource_data in enumerate(rec_topic_data['resources']):
                            TopicResource.objects.create(
                                recommended_topic=recommended_topic,
                                title=resource_data['title'],
                                description=resource_data['description'],
                                url=resource_data['url'],
                                resource_type=resource_data['type'],
                                difficulty=resource_data['difficulty'],
                                estimated_time=resource_data['estimated_time'],
                                order=res_order
                            )
                    except Topic.DoesNotExist:
                        self.stdout.write(f"   ⚠️ Topic not found: {rec_topic_data['topic']}")
                
                self.stdout.write(f"   💾 ✅ Learning path saved to database (ID: {path.id})")
                return path.id
                
        except Exception as e:
            self.stdout.write(f"   💾 ❌ Error saving to database: {e}")
            return None
