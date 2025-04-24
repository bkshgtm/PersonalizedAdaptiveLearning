import os
import django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'pal_project.settings')
django.setup()

from knowledge_graph.models import KnowledgeGraph

def check_ksg():
    count = KnowledgeGraph.objects.count()
    print(f"Knowledge Graph Nodes: {count}")
    if count > 0:
        print("Sample Nodes:", list(KnowledgeGraph.objects.values_list('name', flat=True)[:5]))

if __name__ == '__main__':
    check_ksg()
