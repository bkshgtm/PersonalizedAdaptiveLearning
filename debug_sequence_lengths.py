import os
import django
from collections import defaultdict

# Setup Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'pal_project.settings')
django.setup()

from ml_models.ml.data_preparation import DataPreparation

def analyze_sequences():
    prep = DataPreparation('CS206')
    interactions = prep.get_interactions()
    
    # Count interactions per student
    seq_lengths = defaultdict(int)
    for i in interactions:
        seq_lengths[i['student_id']] += 1
    
    print(f"Total students: {len(seq_lengths)}")
    print(f"Sequence lengths: {sorted(seq_lengths.values())}")
    print(f"Min length: {min(seq_lengths.values())}")
    print(f"Max length: {max(seq_lengths.values())}")
    print(f"Avg length: {sum(seq_lengths.values())/len(seq_lengths):.1f}")

if __name__ == '__main__':
    analyze_sequences()
