import logging
import datetime
from typing import Dict, List, Tuple, Any, Optional
from django.utils import timezone
from django.db import transaction

from core.models import Student, Course, Topic, Resource
from ml_models.models import TopicMastery, PredictionBatch
from knowledge_graph.models import KnowledgeGraph
from knowledge_graph.services.graph_operations import GraphOperations
from learning_paths.models import (
    PathGenerator, PathGenerationJob, LearningPath, 
    LearningPathItem, LearningResource, PathCheckpoint
)

logger = logging.getLogger(__name__)


class LearningPathGenerator:
    """Service for generating personalized learning paths."""
    
    def __init__(self, job_id: int):
        """
        Initialize with a PathGenerationJob ID.
        
        Args:
            job_id: ID of the PathGenerationJob
        """
        self.job = PathGenerationJob.objects.get(pk=job_id)
        self.generator = self.job.generator
        self.student = self.job.student
        self.course = self.job.course
        self.prediction_batch = self.job.prediction_batch
        self.knowledge_graph = self.job.knowledge_graph
        self.config = self.generator.config
        
        # If no knowledge graph is specified, use the active one
        if not self.knowledge_graph:
            try:
                self.knowledge_graph = KnowledgeGraph.objects.get(is_active=True)
            except KnowledgeGraph.DoesNotExist:
                logger.warning("No active knowledge graph found.")
                self.knowledge_graph = None
    
    def generate_path(self) -> LearningPath:
        """
        Generate a learning path for the student.
        
        Returns:
            The generated LearningPath instance
        """
        # Update job status
        self.job.status = 'processing'
        self.job.started_at = timezone.now()
        self.job.save(update_fields=['status', 'started_at'])
        
        try:
            # Get topic masteries
            masteries = self._get_topic_masteries()
            
            # Get knowledge graph
            graph_ops = None
            if self.knowledge_graph:
                graph_ops = GraphOperations(self.knowledge_graph.id)
            
            # Generate the learning path items
            path_items = self._generate_path_items(masteries, graph_ops)
            
            # Create the learning path with transaction
            with transaction.atomic():
                # Create the path
                learning_path = LearningPath.objects.create(
                    generation_job=self.job,
                    student=self.student,
                    course=self.course,
                    name=f"Learning Path for {self.course.title}",
                    description=f"Generated on {timezone.now().strftime('%Y-%m-%d')}",
                    expires_at=timezone.now() + datetime.timedelta(days=30),  # Expire in 30 days
                    overall_progress={
                        'completed_topics': 0,
                        'in_progress_topics': 0,
                        'not_started_topics': len(path_items),
                        'overall_mastery': self._calculate_overall_mastery(masteries)
                    },
                    estimated_completion_time=datetime.timedelta(
                        minutes=sum(item['estimated_review_time'] for item in path_items)
                    )
                )
                
                # Create path items
                self._create_path_items(learning_path, path_items)
                
                # Create checkpoints
                self._create_checkpoints(learning_path, path_items)
                
                # Update job status
                self.job.status = 'completed'
                self.job.completed_at = timezone.now()
                self.job.save(update_fields=['status', 'completed_at'])
                
                return learning_path
                
        except Exception as e:
            logger.exception(f"Error generating learning path: {str(e)}")
            
            # Update job status
            self.job.status = 'failed'
            self.job.error_message = str(e)
            self.job.completed_at = timezone.now()
            self.job.save(update_fields=['status', 'error_message', 'completed_at'])
            
            raise
    
    def _get_topic_masteries(self) -> Dict[int, TopicMastery]:
        """
        Get topic mastery information for the student.
        
        Returns:
            Dictionary mapping topic IDs to TopicMastery instances
        """
        # If prediction batch is specified, use it
        if self.prediction_batch:
            masteries = TopicMastery.objects.filter(
                student=self.student,
                prediction_batch=self.prediction_batch,
                topic__course=self.course
            ).select_related('topic')
        else:
            # Otherwise, use the latest prediction batch
            latest_batch = PredictionBatch.objects.filter(
                status='completed',
                model__course=self.course
            ).order_by('-completed_at').first()
            
            if not latest_batch:
                logger.warning(f"No prediction batch found for course {self.course.course_id}")
                # Return empty dict, default values will be used
                return {}
            
            # Get masteries from the latest batch
            masteries = TopicMastery.objects.filter(
                student=self.student,
                prediction_batch=latest_batch,
                topic__course=self.course
            ).select_related('topic')
        
        # Convert to dict for faster lookup
        mastery_dict = {m.topic.id: m for m in masteries}
        return mastery_dict
    
    def _generate_path_items(
        self, 
        masteries: Dict[int, TopicMastery],
        graph_ops: Optional[GraphOperations] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate a list of path items for the student.
        
        Args:
            masteries: Dictionary mapping topic IDs to TopicMastery instances
            graph_ops: Optional GraphOperations instance
            
        Returns:
            List of dictionaries with path item data
        """
        # Get all topics for the course
        topics = Topic.objects.filter(course=self.course)
        total_topics = topics.count()
        logger.info(f"Course has {total_topics} total topics")
        
        # Get configuration parameters with sensible defaults
        include_strong = self.config.get('include_strong_topics', False)
        max_weak_topics = self.config.get('max_weak_topics', 10)
        max_developing_topics = self.config.get('max_developing_topics', 10)
        max_strong_topics = self.config.get('max_strong_topics', 5)
        
        # Adaptive sizing based on course size
        if total_topics > 100:
            # For very large courses, scale up the limits
            max_weak_topics = min(20, total_topics // 5)
            max_developing_topics = min(15, total_topics // 7)
            max_strong_topics = min(10, total_topics // 10)
        elif total_topics < 20:
            # For small courses, include most topics
            max_weak_topics = total_topics
            max_developing_topics = total_topics
            max_strong_topics = total_topics if include_strong else 0
        
        # Process each topic
        weak_items = []
        developing_items = []
        strong_items = []
        
        for topic in topics:
            # Get mastery for this topic, or use default values
            mastery = masteries.get(topic.id)
            
            if mastery:
                proficiency_score = mastery.mastery_score
                trend = mastery.trend
                confidence = mastery.confidence
            else:
                # Default values for topics without mastery data
                proficiency_score = 0.5  # Neutral
                trend = 'stagnant'
                confidence = 0.5
            
            # Determine status based on proficiency
            if proficiency_score < 0.4:
                status = 'weak'
            elif proficiency_score < 0.7:
                status = 'developing'
            else:
                status = 'strong'
            
            # Skip strong topics if configured to exclude them
            if status == 'strong' and not include_strong:
                continue
            
            # Get recommended resources
            recommended_resources = self._get_recommended_resources(topic, proficiency_score)
            
            # Calculate estimated review time
            estimated_review_time = sum(r['estimated_time'] for r in recommended_resources)
            
            # Generate reason for recommendation
            reason = self._generate_recommendation_reason(topic, proficiency_score, trend)
            
            # Calculate priority
            priority = self._calculate_priority(topic, proficiency_score, trend, graph_ops)
            
            # Create item data
            item_data = {
                'topic': topic,
                'proficiency_score': proficiency_score,
                'status': status,
                'trend': trend,
                'confidence_of_improvement': 0.8 if trend == 'improving' else 0.6,
                'reason': reason,
                'estimated_review_time': estimated_review_time,
                'priority': priority,
                'recommended_resources': recommended_resources
            }
            
            # Add to appropriate category
            if status == 'weak':
                weak_items.append(item_data)
            elif status == 'developing':
                developing_items.append(item_data)
            else:  # strong
                strong_items.append(item_data)
        
        # Sort each category by priority
        weak_items.sort(key=lambda x: x['priority'])
        developing_items.sort(key=lambda x: x['priority'])
        strong_items.sort(key=lambda x: x['priority'])
        
        # Take top N from each category
        weak_items = weak_items[:max_weak_topics]
        developing_items = developing_items[:max_developing_topics]
        strong_items = strong_items[:max_strong_topics]
        
        # Combine all items
        path_items = weak_items + developing_items + strong_items
        
        # Final sort by priority
        path_items.sort(key=lambda x: x['priority'])
        
        # Log summary
        logger.info(f"Generated path with {len(path_items)} items: "
                   f"{len(weak_items)} weak, {len(developing_items)} developing, "
                   f"{len(strong_items)} strong")
        
        return path_items
    
    def _calculate_priority(
        self, 
        topic: Topic, 
        proficiency_score: float, 
        trend: str,
        graph_ops: Optional[GraphOperations] = None
    ) -> int:
        """
        Calculate priority for a topic in the learning path using enhanced analysis.
        
        Args:
            topic: Topic instance
            proficiency_score: Mastery score (0-1)
            trend: Trend of mastery ('improving', 'declining', 'stagnant')
            graph_ops: Optional GraphOperations instance
            
        Returns:
            Priority value (lower is higher priority)
        """
        # Base priority is inversely proportional to proficiency
        base_priority = int((1 - proficiency_score) * 100)
        
        # Adjust priority based on trend
        if trend == 'declining':
            # Declining topics get higher priority (lower value)
            base_priority -= 20
        elif trend == 'improving':
            # Improving topics get lower priority (higher value)
            base_priority += 10
        
        # If we have enhanced graph operations, use detailed analysis
        if graph_ops and hasattr(graph_ops, 'get_detailed_prerequisites'):
            try:
                # Get current student masteries
                student_masteries = {}
                if self.prediction_batch:
                    masteries = TopicMastery.objects.filter(
                        student=self.student,
                        prediction_batch=self.prediction_batch
                    )
                    student_masteries = {m.topic_id: m.mastery_score for m in masteries}
                
                # Get detailed prerequisite analysis
                prereq_analysis = graph_ops.get_detailed_prerequisites(topic.id, student_masteries)
                
                # Adjust priority based on prerequisite strength
                prereq_strength = prereq_analysis.get('prerequisite_strength', 0.0)
                missing_count = len(prereq_analysis.get('missing_prereqs', []))
                
                # Higher prerequisite strength = higher priority (lower value)
                base_priority -= int(prereq_strength * 20)
                
                # More missing prerequisites = lower priority (higher value)
                base_priority += missing_count * 15
                
                # Boost priority for topics with many fine-grained prerequisites
                fine_grained_count = len(prereq_analysis.get('fine_grained_prereqs', []))
                if fine_grained_count > len(prereq_analysis.get('topic_level_prereqs', [])):
                    # This topic has more detailed prerequisites, boost its priority
                    base_priority -= 10
                
            except Exception as e:
                logger.warning(f"Error in enhanced priority calculation: {str(e)}")
                # Fall back to basic calculation
                self._calculate_basic_priority(topic, graph_ops, base_priority)
        elif graph_ops:
            # Fall back to basic graph-based priority calculation
            base_priority = self._calculate_basic_priority(topic, graph_ops, base_priority)
        
        # Ensure priority is within reasonable bounds
        base_priority = max(1, min(base_priority, 999))
        
        return base_priority
    
    def _calculate_basic_priority(self, topic: Topic, graph_ops: GraphOperations, base_priority: int) -> int:
        """Basic priority calculation for fallback."""
        try:
            next_topics = graph_ops.get_next_topics(topic.id)
            if next_topics:
                # The more topics this is a prerequisite for, the higher the priority
                base_priority -= len(next_topics) * 5
            
            # Topics with unmet prerequisites get lower priority
            prerequisites = graph_ops.get_prerequisites(topic.id, direct_only=True)
            if prerequisites:
                # Get mastery for prerequisites
                prereq_masteries = TopicMastery.objects.filter(
                    student=self.student,
                    prediction_batch=self.prediction_batch,
                    topic_id__in=prerequisites
                )
                
                # Calculate average mastery of prerequisites
                if prereq_masteries:
                    prereq_mastery = sum(m.mastery_score for m in prereq_masteries) / len(prereq_masteries)
                    # If prerequisites are not well mastered, lower priority of this topic
                    if prereq_mastery < 0.6:
                        base_priority += 30
        except Exception as e:
            logger.warning(f"Error calculating basic graph-based priority: {str(e)}")
        
        return base_priority
    
    def _get_recommended_resources(
        self, 
        topic: Topic, 
        proficiency_score: float
    ) -> List[Dict[str, Any]]:
        """
        Get recommended resources for a topic.
        
        Args:
            topic: Topic instance
            proficiency_score: Mastery score (0-1)
            
        Returns:
            List of dictionaries with resource data
        """
        # Get resources for this topic
        resources = Resource.objects.filter(topics=topic)
        
        if not resources:
            logger.warning(f"No resources found for topic {topic.name} (ID: {topic.id})")
            return []
        
        logger.info(f"Found {resources.count()} resources for topic {topic.name}")
        
        # Use the ResourceSelector service for better resource selection
        from learning_paths.services.resource_selector import ResourceSelector
        
        selector = ResourceSelector(config=self.config)
        max_resources = self.config.get('max_resources_per_topic', 3)
        
        recommended = selector.select_resources(
            topic=topic,
            proficiency_score=proficiency_score,
            max_resources=max_resources
        )
        
        if not recommended:
            logger.warning(f"No resources selected for topic {topic.name} - falling back to default selection")
            # Process resources with the old method as fallback
            for resource in resources:
                # Determine if resource difficulty matches student's needs
                difficulty_match = False
                
                if proficiency_score < 0.4 and resource.difficulty == 'beginner':
                    difficulty_match = True
                    match_reason = "Beginner resource for a topic you're still learning"
                elif 0.4 <= proficiency_score < 0.7 and resource.difficulty == 'intermediate':
                    difficulty_match = True
                    match_reason = "Intermediate resource to help you progress further"
                elif proficiency_score >= 0.7 and resource.difficulty == 'advanced':
                    difficulty_match = True
                    match_reason = "Advanced resource to master this topic"
                else:
                    # If no perfect match, still include some resources
                    difficulty_match = True
                    match_reason = "Resource to build your understanding of this topic"
                
                if difficulty_match:
                    # Convert duration to minutes
                    if resource.estimated_time:
                        minutes = resource.estimated_time.total_seconds() // 60
                    else:
                        minutes = 30  # Default 30 minutes
                    
                    recommended.append({
                        'resource': resource,
                        'match_reason': match_reason,
                        'estimated_time': minutes
                    })
            
            # Sort by resource type to get a mix
            recommended.sort(key=lambda x: x['resource'].resource_type)
            
            # Limit to max resources per topic
            recommended = recommended[:max_resources]
        
        logger.info(f"Selected {len(recommended)} resources for topic {topic.name}")
        return recommended
    
    def _generate_recommendation_reason(
        self, 
        topic: Topic, 
        proficiency_score: float, 
        trend: str
    ) -> str:
        """
        Generate a reason for recommending a topic.
        
        Args:
            topic: Topic instance
            proficiency_score: Mastery score (0-1)
            trend: Trend of mastery ('improving', 'declining', 'stagnant')
            
        Returns:
            Reason text
        """
        if proficiency_score < 0.4:
            if trend == 'declining':
                return f"Your understanding of {topic.name} is declining and needs immediate attention."
            elif trend == 'stagnant':
                return f"You've been struggling with {topic.name} consistently. Focus on building your foundation."
            else:  # improving
                return f"You're making progress with {topic.name}, but still need more practice."
        elif proficiency_score < 0.7:
            if trend == 'declining':
                return f"Your grasp of {topic.name} has been slipping recently. Let's reinforce it."
            elif trend == 'stagnant':
                return f"You have a basic understanding of {topic.name}, but need to deepen your knowledge."
            else:  # improving
                return f"You're steadily improving in {topic.name}. Keep building on this progress."
        else:
            if trend == 'declining':
                return f"Your mastery of {topic.name} is high but beginning to decline. Let's review it."
            elif trend == 'stagnant':
                return f"You have a strong grasp of {topic.name}. This is a recommended review to maintain your knowledge."
            else:  # improving
                return f"You're doing very well with {topic.name}. This is an opportunity to fully master the topic."
    
    def _create_path_items(self, learning_path: LearningPath, path_items: List[Dict[str, Any]]) -> None:
        """
        Create LearningPathItem instances for a learning path.
        
        Args:
            learning_path: LearningPath instance
            path_items: List of dictionaries with path item data
        """
        for i, item_data in enumerate(path_items, 1):
            # Create path item
            path_item = LearningPathItem.objects.create(
                path=learning_path,
                topic=item_data['topic'],
                priority=i,  # Use the sorted order as priority
                status=item_data['status'],
                proficiency_score=item_data['proficiency_score'],
                trend=item_data['trend'],
                confidence_of_improvement=item_data['confidence_of_improvement'],
                reason=item_data['reason'],
                estimated_review_time=datetime.timedelta(minutes=item_data['estimated_review_time']),
                completed=False
            )
            
            # Create resources for the path item
            for res_data in item_data['recommended_resources']:
                LearningResource.objects.create(
                    path_item=path_item,
                    resource=res_data['resource'],
                    match_reason=res_data['match_reason'],
                    viewed=False
                )
    
    def _create_checkpoints(self, learning_path: LearningPath, path_items: List[Dict[str, Any]]) -> None:
        """
        Create checkpoints for a learning path.
        
        Args:
            learning_path: LearningPath instance
            path_items: List of dictionaries with path item data
        """
        # Group items by status
        status_groups = {
            'weak': [],
            'developing': [],
            'strong': []
        }
        
        for item in path_items:
            status = item['status']
            if status in status_groups:
                status_groups[status].append(item)
        
        # Create checkpoints for each group
        position = 1
        
        # Checkpoint for weak topics
        if status_groups['weak']:
            weak_topics = [item['topic'] for item in status_groups['weak']]
            if weak_topics:
                checkpoint = PathCheckpoint.objects.create(
                    path=learning_path,
                    name="Foundation Checkpoint",
                    description="Test your understanding of fundamental topics",
                    checkpoint_type='quiz',
                    position=position,
                    completed=False
                )
                checkpoint.topics.set(weak_topics[:5])  # First 5 weak topics
                position += 1
        
        # Checkpoint for developing topics
        if status_groups['developing']:
            dev_topics = [item['topic'] for item in status_groups['developing']]
            if dev_topics:
                checkpoint = PathCheckpoint.objects.create(
                    path=learning_path,
                    name="Progress Checkpoint",
                    description="Assess your growing knowledge of these topics",
                    checkpoint_type='exercise',
                    position=position,
                    completed=False
                )
                checkpoint.topics.set(dev_topics[:5])  # First 5 developing topics
                position += 1
        
        # Final checkpoint
        topics = [item['topic'] for item in path_items[:10]]  # Top 10 priority topics
        if topics:
            checkpoint = PathCheckpoint.objects.create(
                path=learning_path,
                name="Mastery Checkpoint",
                description="Demonstrate your comprehensive understanding",
                checkpoint_type='project',
                position=position,
                completed=False
            )
            checkpoint.topics.set(topics)
    
    def _calculate_overall_mastery(self, masteries: Dict[int, TopicMastery]) -> float:
        """
        Calculate overall mastery score across all topics.
        
        Args:
            masteries: Dictionary mapping topic IDs to TopicMastery instances
            
        Returns:
            Overall mastery score (0-1)
        """
        if not masteries:
            return 0.5  # Default for no data
        
        # Calculate average mastery
        total_mastery = sum(m.mastery_score for m in masteries.values())
        return total_mastery / len(masteries)
