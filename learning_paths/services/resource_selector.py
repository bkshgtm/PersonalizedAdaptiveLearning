import logging
from typing import Dict, List, Any, Optional

from core.models import Topic, Resource
from ml_models.models import TopicMastery

logger = logging.getLogger(__name__)


class ResourceSelector:
    """Service for selecting appropriate learning resources for students."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize with optional configuration.
        
        Args:
            config: Optional configuration parameters
        """
        self.config = config or {}
    
    def select_resources(
        self, 
        topic: Topic, 
        proficiency_score: float,
        learning_style: Optional[str] = None,
        max_resources: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Select appropriate resources for a student based on their proficiency.
        
        Args:
            topic: Topic to select resources for
            proficiency_score: Student's proficiency score for this topic (0-1)
            learning_style: Optional learning style preference
            max_resources: Maximum number of resources to return
            
        Returns:
            List of dictionaries with resource data
        """
        # Get all resources for this topic
        resources = Resource.objects.filter(topics=topic)
        
        if not resources:
            logger.warning(f"No resources found for topic {topic.name}")
            return []
        
        # Score each resource based on criteria
        scored_resources = []
        
        for resource in resources:
            score = self._score_resource(resource, proficiency_score, learning_style)
            
            # Convert duration to minutes for consistency
            if resource.estimated_time:
                minutes = resource.estimated_time.total_seconds() // 60
            else:
                minutes = 30  # Default 30 minutes
            
            # Generate match reason
            match_reason = self._generate_match_reason(resource, proficiency_score, learning_style)
            
            scored_resources.append({
                'resource': resource,
                'score': score,
                'match_reason': match_reason,
                'estimated_time': minutes
            })
        
        # Sort by score (highest first)
        scored_resources.sort(key=lambda x: x['score'], reverse=True)
        
        # Also ensure diversity of resource types
        selected_resources = []
        selected_types = set()
        
        # First pass: get at least one of each type if available
        for resource_data in scored_resources:
            resource = resource_data['resource']
            
            if resource.resource_type not in selected_types and len(selected_resources) < max_resources:
                selected_resources.append(resource_data)
                selected_types.add(resource.resource_type)
        
        # Second pass: fill remaining slots with highest scored resources
        for resource_data in scored_resources:
            if resource_data not in selected_resources and len(selected_resources) < max_resources:
                selected_resources.append(resource_data)
                
            if len(selected_resources) >= max_resources:
                break
        
        return selected_resources
    
    def _score_resource(
        self, 
        resource: Resource, 
        proficiency_score: float,
        learning_style: Optional[str] = None
    ) -> float:
        """
        Score a resource based on how well it matches the student's needs.
        
        Args:
            resource: Resource to score
            proficiency_score: Student's proficiency score (0-1)
            learning_style: Optional learning style preference
        
        Returns:
            Score value (higher is better match)
        """
        score = 0.0
        
        # Score based on difficulty match
        if proficiency_score < 0.4:  # Beginner level
            if resource.difficulty == 'beginner':
                score += 10.0
            elif resource.difficulty == 'intermediate':
                score += 5.0
            # Advanced resources get no bonus for beginners
        elif proficiency_score < 0.7:  # Intermediate level
            if resource.difficulty == 'intermediate':
                score += 10.0
            elif resource.difficulty == 'beginner':
                score += 7.0
            elif resource.difficulty == 'advanced':
                score += 3.0
        else:  # Advanced level
            if resource.difficulty == 'advanced':
                score += 10.0
            elif resource.difficulty == 'intermediate':
                score += 7.0
            # Beginner resources get no bonus for advanced students
        
        # Always give some base score to ensure resources are selected
        # even if they don't match the ideal difficulty
        score += 1.0
        
        # Score based on resource type (configurable weights)
        type_weights = self.config.get('resource_type_weights', {})
        default_weight = 1.0
        score += type_weights.get(resource.resource_type, default_weight)
        
        # Score inversely based on estimated time if the student has limited time
        time_sensitivity = self.config.get('time_sensitivity', 0.0)  # 0.0 = no time constraint
        if time_sensitivity > 0 and resource.estimated_time:
            # Shorter resources get higher scores when time is limited
            minutes = resource.estimated_time.total_seconds() // 60
            time_factor = max(0, 1.0 - (minutes / 120.0))  # Normalize to 0-1, 2 hours as max
            score += time_factor * time_sensitivity * 5.0
        
        return score
    
    def _generate_match_reason(
        self, 
        resource: Resource, 
        proficiency_score: float,
        learning_style: Optional[str] = None
    ) -> str:
        """
        Generate a reason explaining why this resource was selected.
        
        Args:
            resource: Resource to explain
            proficiency_score: Student's proficiency score (0-1)
            learning_style: Optional learning style preference
            
        Returns:
            Explanation text
        """
        reasons = []
        
        # Reason based on difficulty
        if proficiency_score < 0.4 and resource.difficulty == 'beginner':
            reasons.append("This beginner resource is perfect for building your foundation")
        elif 0.4 <= proficiency_score < 0.7 and resource.difficulty == 'intermediate':
            reasons.append("This intermediate resource matches your current understanding")
        elif proficiency_score >= 0.7 and resource.difficulty == 'advanced':
            reasons.append("This advanced resource will help you master the topic")
        elif proficiency_score < 0.5 and resource.difficulty == 'beginner':
            reasons.append("This beginner-friendly resource will strengthen your understanding")
        elif proficiency_score >= 0.5 and resource.difficulty == 'intermediate':
            reasons.append("This resource will help expand your knowledge")
        elif proficiency_score >= 0.7 and resource.difficulty == 'intermediate':
            reasons.append("This provides good review of the topic")
        
        # Reason based on resource type
        if resource.resource_type == 'video':
            reasons.append("Video content for visual learners")
        elif resource.resource_type == 'document':
            reasons.append("Comprehensive documentation")
        elif resource.resource_type == 'exercise':
            reasons.append("Hands-on practice to reinforce concepts")
        elif resource.resource_type == 'tutorial':
            reasons.append("Step-by-step tutorial to guide your learning")
        elif resource.resource_type == 'quiz':
            reasons.append("Test your knowledge with this assessment")
        
        # Reason based on learning style
        if learning_style and hasattr(resource, 'learning_style') and resource.learning_style == learning_style:
            reasons.append(f"Matches your {learning_style} learning style")
        
        # Combine reasons
        if reasons:
            return ". ".join(reasons) + "."
        else:
            return "This resource covers important content for this topic."
