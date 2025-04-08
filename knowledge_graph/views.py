from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.http import JsonResponse
from django.views.decorators.http import require_POST

from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework import status

from .models import KnowledgeGraph, GraphEdge
from .services.graph_operations import GraphOperations
from .services.graph_visualizer import GraphVisualizer
from core.models import Topic


@login_required
def graph_list(request):
    """
    View to show all knowledge graphs.
    """
    graphs = KnowledgeGraph.objects.all().order_by('-updated_at')
    return render(request, 'knowledge_graph/graph_list.html', {'graphs': graphs})


@login_required
def graph_detail(request, graph_id):
    """
    View to show details of a knowledge graph.
    """
    graph = get_object_or_404(KnowledgeGraph, pk=graph_id)
    
    # Get graph operations service
    graph_ops = GraphOperations(graph.id)
    visualizer = GraphVisualizer(graph_ops)
    
    # Get visualization data
    vis_data = visualizer.get_vis_js_format()
    
    # Get sample path for visualization
    # Get sample path using the updated method
    sample_path = graph_ops.get_sample_path()
    
    return render(request, 'knowledge_graph/graph_detail.html', {
        'graph': graph,
        'vis_data': vis_data,
        'sample_path': sample_path,
        'graph_json': graph.data
    })


@login_required
def create_graph(request):
    """
    View to create a new knowledge graph.
    """
    if request.method == 'POST':
        name = request.POST.get('name')
        description = request.POST.get('description', '')
        version = request.POST.get('version', '1.0')
        is_active = request.POST.get('is_active') == 'on'
        
        if not name:
            messages.error(request, "Graph name is required.")
            return redirect('create_graph')
        
        # If this graph is set as active, unset other active graphs
        if is_active:
            KnowledgeGraph.objects.filter(is_active=True).update(is_active=False)
        
        # Create an empty graph structure
        graph_data = {
            'nodes': [],
            'edges': []
        }
        
        # Create the graph
        graph = KnowledgeGraph.objects.create(
            name=name,
            description=description,
            version=version,
            created_by=request.user,
            data=graph_data,
            is_active=is_active
        )
        
        messages.success(request, f"Knowledge graph '{name}' created successfully.")
        return redirect('graph_detail', graph_id=graph.id)
    
    return render(request, 'knowledge_graph/create_graph.html')


@require_POST
@login_required
def set_active_graph(request, graph_id):
    """
    View to set a graph as active.
    """
    graph = get_object_or_404(KnowledgeGraph, pk=graph_id)
    
    # Unset other active graphs
    KnowledgeGraph.objects.filter(is_active=True).update(is_active=False)
    
    # Set this graph as active
    graph.is_active = True
    graph.save(update_fields=['is_active'])
    
    messages.success(request, f"Graph '{graph.name}' is now the active graph.")
    return redirect('graph_list')


@login_required
def upload_graph(request):
    """
    View to upload a JSON graph.
    """
    if request.method == 'POST':
        name = request.POST.get('name')
        description = request.POST.get('description', '')
        version = request.POST.get('version', '1.0')
        is_active = request.POST.get('is_active') == 'on'
        
        if not name:
            messages.error(request, "Graph name is required.")
            return redirect('upload_graph')
        
        if 'graph_file' not in request.FILES:
            messages.error(request, "Please upload a JSON file.")
            return redirect('upload_graph')
        
        # Read the JSON file
        try:
            import json
            graph_file = request.FILES['graph_file']
            graph_data = json.loads(graph_file.read().decode('utf-8'))
            
            # Validate the JSON structure
            if 'nodes' not in graph_data or 'edges' not in graph_data:
                messages.error(request, "Invalid graph format. JSON must contain 'nodes' and 'edges' arrays.")
                return redirect('upload_graph')
        except Exception as e:
            messages.error(request, f"Error parsing JSON file: {str(e)}")
            return redirect('upload_graph')
        
        # If this graph is set as active, unset other active graphs
        if is_active:
            KnowledgeGraph.objects.filter(is_active=True).update(is_active=False)
        
        # Create the graph
        graph = KnowledgeGraph.objects.create(
            name=name,
            description=description,
            version=version,
            created_by=request.user,
            data=graph_data,
            is_active=is_active
        )
        
        # Initialize graph operations to create edges
        graph_ops = GraphOperations(graph.id)
        graph_ops.from_json(graph_data)
        
        messages.success(request, f"Knowledge graph '{name}' uploaded successfully.")
        return redirect('graph_detail', graph_id=graph.id)
    
    return render(request, 'knowledge_graph/upload_graph.html')


@login_required
def topic_detail(request, graph_id, topic_id):
    """
    View to show topic details in the context of a knowledge graph.
    """
    graph = get_object_or_404(KnowledgeGraph, pk=graph_id)
    topic = get_object_or_404(Topic, pk=topic_id)
    
    # Get graph operations service
    graph_ops = GraphOperations(graph.id)
    
    # Get topic relationships
    prerequisites = graph_ops.get_prerequisites(topic.id)
    next_topics = graph_ops.get_next_topics(topic.id)
    related_topics = graph_ops.get_related_topics(topic.id)
    
    return render(request, 'knowledge_graph/topic_detail.html', {
        'graph': graph,
        'topic': topic,
        'prerequisites': prerequisites,
        'next_topics': next_topics,
        'related_topics': related_topics
    })


@login_required
def add_edge(request, graph_id):
    """
    View to add an edge to the graph.
    """
    if request.method == 'POST':
        source_id = request.POST.get('source_id')
        target_id = request.POST.get('target_id')
        relationship = request.POST.get('relationship')
        weight = request.POST.get('weight', 1.0)
        
        if not source_id or not target_id or not relationship:
            messages.error(request, "Source, target, and relationship are required.")
            return redirect('graph_detail', graph_id=graph_id)
        
        try:
            # Convert to proper types
            source_id = int(source_id)
            target_id = int(target_id)
            weight = float(weight)
            
            # Validate that topics exist
            source_topic = Topic.objects.get(pk=source_id)
            target_topic = Topic.objects.get(pk=target_id)
            
            # Get the graph
            graph = KnowledgeGraph.objects.get(pk=graph_id)
            
            # Add the edge
            edge = GraphEdge.objects.create(
                graph=graph,
                source_topic=source_topic,
                target_topic=target_topic,
                relationship_type=relationship,
                weight=weight
            )
            
            # Update the graph's JSON
            graph_ops = GraphOperations(graph.id)
            graph_ops.save_graph()
            
            messages.success(request, f"Edge from '{source_topic.name}' to '{target_topic.name}' added successfully.")
        except (Topic.DoesNotExist, KnowledgeGraph.DoesNotExist) as e:
            messages.error(request, f"Error: {str(e)}")
        except Exception as e:
            messages.error(request, f"Error adding edge: {str(e)}")
        
        return redirect('graph_detail', graph_id=graph_id)
    
    # GET request: show the form
    graph = get_object_or_404(KnowledgeGraph, pk=graph_id)
    topics = Topic.objects.all()
    
    return render(request, 'knowledge_graph/add_edge.html', {
        'graph': graph,
        'topics': topics,
        'relationship_types': GraphEdge.RELATIONSHIP_TYPES
    })


@login_required
def remove_edge(request, edge_id):
    """
    View to remove an edge from the graph.
    """
    edge = get_object_or_404(GraphEdge, pk=edge_id)
    graph_id = edge.graph.id
    
    # Delete the edge
    edge.delete()
    
    # Update the graph's JSON
    graph_ops = GraphOperations(graph_id)
    graph_ops.save_graph()
    
    messages.success(request, "Edge removed successfully.")
    return redirect('graph_detail', graph_id=graph_id)


@login_required
def export_graph(request, graph_id):
    """
    View to export the graph as JSON.
    """
    graph = get_object_or_404(KnowledgeGraph, pk=graph_id)
    
    # Return the graph data as JSON
    return JsonResponse(graph.data)


@login_required
def learning_path(request, graph_id):
    """
    View to visualize a learning path.
    """
    if request.method == 'POST':
        start_topics = request.POST.getlist('start_topics')
        target_topics = request.POST.getlist('target_topics')
        
        if not start_topics or not target_topics:
            messages.error(request, "Start and target topics are required.")
            return redirect('learning_path', graph_id=graph_id)
        
        # Convert to integers
        start_topics = [int(topic_id) for topic_id in start_topics]
        target_topics = [int(topic_id) for topic_id in target_topics]
        
        # Get the graph
        graph = get_object_or_404(KnowledgeGraph, pk=graph_id)
        
        # Get graph operations
        graph_ops = GraphOperations(graph.id)
        visualizer = GraphVisualizer(graph_ops)
        
        # Get the learning path visualization
        vis_data = visualizer.get_learning_path_visualization(start_topics, target_topics)
        
        # Get the path as a list
        path = graph_ops.find_learning_path(start_topics, target_topics)
        
        return render(request, 'knowledge_graph/learning_path_result.html', {
            'graph': graph,
            'vis_data': vis_data,
            'path': path,
            'start_topics': Topic.objects.filter(pk__in=start_topics),
            'target_topics': Topic.objects.filter(pk__in=target_topics)
        })
    
    # GET request: show the form
    graph = get_object_or_404(KnowledgeGraph, pk=graph_id)
    topics = Topic.objects.all()
    
    return render(request, 'knowledge_graph/learning_path_form.html', {
        'graph': graph,
        'topics': topics
    })


# API Views
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_active_graph_api(request):
    """
    API endpoint to get the active knowledge graph.
    """
    try:
        graph = KnowledgeGraph.objects.get(is_active=True)
    except KnowledgeGraph.DoesNotExist:
        return Response(
            {"error": "No active knowledge graph found."},
            status=status.HTTP_404_NOT_FOUND
        )
    
    # Get the graph operations service
    graph_ops = GraphOperations(graph.id)
    
    # Get graph data
    graph_data = graph_ops.to_json()
    
    return Response({
        "id": graph.id,
        "name": graph.name,
        "version": graph.version,
        "data": graph_data
    })


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def upload_graph_api(request):
    """
    API endpoint to upload a knowledge graph.
    """
    name = request.data.get('name')
    description = request.data.get('description', '')
    version = request.data.get('version', '1.0')
    is_active = request.data.get('is_active', False)
    graph_data = request.data.get('data')
    
    if not name:
        return Response(
            {"error": "Graph name is required."},
            status=status.HTTP_400_BAD_REQUEST
        )
    
    if not graph_data:
        return Response(
            {"error": "Graph data is required."},
            status=status.HTTP_400_BAD_REQUEST
        )
    
    # Validate the graph data structure
    if 'nodes' not in graph_data or 'edges' not in graph_data:
        return Response(
            {"error": "Invalid graph format. JSON must contain 'nodes' and 'edges' arrays."},
            status=status.HTTP_400_BAD_REQUEST
        )
    
    # If this graph is set as active, unset other active graphs
    if is_active:
        KnowledgeGraph.objects.filter(is_active=True).update(is_active=False)
    
    # Create the graph
    graph = KnowledgeGraph.objects.create(
        name=name,
        description=description,
        version=version,
        created_by=request.user,
        data=graph_data,
        is_active=is_active
    )
    
    # Initialize graph operations to create edges
    graph_ops = GraphOperations(graph.id)
    graph_ops.from_json(graph_data)
    
    return Response({
        "id": graph.id,
        "name": graph.name,
        "version": graph.version,
        "is_active": graph.is_active
    }, status=status.HTTP_201_CREATED)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_prerequisites_api(request, topic_id):
    """
    API endpoint to get prerequisites for a topic.
    """
    # Get the active graph
    try:
        graph = KnowledgeGraph.objects.get(is_active=True)
    except KnowledgeGraph.DoesNotExist:
        return Response(
            {"error": "No active knowledge graph found."},
            status=status.HTTP_404_NOT_FOUND
        )
    
    # Get the graph operations service
    graph_ops = GraphOperations(graph.id)
    
    # Get direct_only parameter
    direct_only = request.query_params.get('direct_only', 'false').lower() == 'true'
    
    # Get prerequisites
    try:
        prerequisites = graph_ops.get_prerequisites(int(topic_id), direct_only=direct_only)
        return Response(prerequisites)
    except Exception as e:
        return Response(
            {"error": str(e)},
            status=status.HTTP_400_BAD_REQUEST
        )


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_next_topics_api(request, topic_id):
    """
    API endpoint to get next topics for a topic.
    """
    # Get the active graph
    try:
        graph = KnowledgeGraph.objects.get(is_active=True)
    except KnowledgeGraph.DoesNotExist:
        return Response(
            {"error": "No active knowledge graph found."},
            status=status.HTTP_404_NOT_FOUND
        )
    
    # Get the graph operations service
    graph_ops = GraphOperations(graph.id)
    
    # Get next topics
    try:
        next_topics = graph_ops.get_next_topics(int(topic_id))
        return Response(next_topics)
    except Exception as e:
        return Response(
            {"error": str(e)},
            status=status.HTTP_400_BAD_REQUEST
        )


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def find_learning_path_api(request):
    """
    API endpoint to find a learning path.
    """
    start_topics = request.data.get('start_topics', [])
    target_topics = request.data.get('target_topics', [])
    
    if not start_topics or not target_topics:
        return Response(
            {"error": "Start and target topics are required."},
            status=status.HTTP_400_BAD_REQUEST
        )
    
    # Get the active graph
    try:
        graph = KnowledgeGraph.objects.get(is_active=True)
    except KnowledgeGraph.DoesNotExist:
        return Response(
            {"error": "No active knowledge graph found."},
            status=status.HTTP_404_NOT_FOUND
        )
    
    # Get the graph operations service
    graph_ops = GraphOperations(graph.id)
    
    # Find the learning path
    try:
        path = graph_ops.find_learning_path(start_topics, target_topics)
        return Response(path)
    except Exception as e:
        return Response(
            {"error": str(e)},
            status=status.HTTP_400_BAD_REQUEST
        )


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def get_available_topics_api(request):
    """
    API endpoint to get available topics based on mastery.
    """
    mastered_topics = request.data.get('mastered_topics', [])
    
    # Get the active graph
    try:
        graph = KnowledgeGraph.objects.get(is_active=True)
    except KnowledgeGraph.DoesNotExist:
        return Response(
            {"error": "No active knowledge graph found."},
            status=status.HTTP_404_NOT_FOUND
        )
    
    # Get the graph operations service
    graph_ops = GraphOperations(graph.id)
    
    # Get available topics
    try:
        available = graph_ops.get_available_topics(mastered_topics)
        blocked = graph_ops.get_blocked_topics(mastered_topics)
        
        return Response({
            "available": available,
            "blocked": blocked
        })
    except Exception as e:
        return Response(
            {"error": str(e)},
            status=status.HTTP_400_BAD_REQUEST
        )
