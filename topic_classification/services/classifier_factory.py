# topic_classification/services/classifier_factory.py
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from topic_classification.services.classifier import TopicClassifier

def get_classifier(model_id: int) -> 'TopicClassifier':
    """
    Factory function to get the appropriate classifier based on model type.
    """
    from topic_classification.models import ClassificationModel
    
    model = ClassificationModel.objects.get(pk=model_id)
    
    if model.model_type == 'tfidf':
        from topic_classification.services.classifier import TfidfClassifier
        return TfidfClassifier(model_id)
    elif model.model_type == 'transformer':
        from topic_classification.services.classifier import TransformerClassifier
        return TransformerClassifier(model_id)
    elif model.model_type in ['openai', 'anthropic']:
        from topic_classification.services.classifier import APIClassifier
        return APIClassifier(model_id)
    elif model.model_type == 'dissect':
        from topic_classification.services.dissect_classifier import DissectClassifier
        return DissectClassifier(model_id)
    else:
        raise ValueError(f"Unsupported model type: {model.model_type}")
