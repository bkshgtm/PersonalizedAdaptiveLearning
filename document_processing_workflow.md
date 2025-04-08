# Document Processing Workflow with DeepSeek Integration

This document provides a visual representation of the document upload and processing workflow, highlighting the DeepSeek integration points for topic classification and answer validation.

## Overall Process Flow

```mermaid
flowchart TD
    A[Upload Document] --> B[Document Stored]
    B --> C[Extract Questions & Answers]
    C --> D[Classify Topics]
    D --> E[Validate Answers]
    E --> F[Update Learning Paths]
    
    C -- DeepSeek API --> C1[AI-based Q&A Extraction]
    C -- Fallback --> C2[Rule-based Extraction]
    
    D -- DeepSeek API --> D1[AI-based Classification]
    D -- Fallback --> D2[Keyword Matching]
    
    E -- DeepSeek API --> E1[AI-based Validation]
    E -- Fallback --> E2[Basic Similarity Check]
```

## Document Upload Process

```mermaid
sequenceDiagram
    actor User
    participant UI as Web Interface
    participant Django as Django Server
    participant Storage as File Storage
    participant DB as Database
    
    User->>UI: Upload document
    UI->>Django: POST /data-ingestion/upload/
    Django->>Storage: Save file
    Django->>DB: Create DocumentUpload record
    Django->>DB: Set status = "pending"
    Django->>User: Redirect to document list
```

## Question Extraction Process

```mermaid
sequenceDiagram
    participant Celery as Celery Worker
    participant DP as DocumentProcessor
    participant DS as DeepSeekClient
    participant DB as Database
    
    Celery->>DP: process()
    DP->>DB: Update status = "processing"
    
    alt PDF Document
        DP->>DP: _process_pdf()
    else DOCX Document
        DP->>DP: _process_docx()
    else Text Document
        DP->>DP: _process_text()
    end
    
    DP->>DP: _extract_qa_from_text()
    DP->>DS: Call DeepSeek API
    
    alt DeepSeek API Success
        DS->>DP: Return extracted Q&A pairs
    else DeepSeek API Failure
        DP->>DP: _simple_qa_extraction() (fallback)
    end
    
    DP->>DB: Save ExtractedQuestion records
```

## Topic Classification Process

```mermaid
sequenceDiagram
    participant DP as DocumentProcessor
    participant CF as ClassifierFactory
    participant DC as DeepSeekClassifier
    participant DS as DeepSeekClient
    participant DB as Database
    
    DP->>CF: get_classifier()
    CF->>DC: Return DeepSeekClassifier
    
    loop For each question
        DP->>DC: classify_question()
        DC->>DS: classify_topic()
        
        alt DeepSeek API Success
            DS->>DC: Return topic_id and confidence
        else DeepSeek API Failure
            DC->>DC: Fallback to keyword matching
        end
        
        DC->>DP: Return Topic and confidence
        DP->>DB: Update ExtractedQuestion with topic
    end
```

## Answer Validation Process

```mermaid
sequenceDiagram
    participant DP as DocumentProcessor
    participant AV as AnswerValidator
    participant DS as DeepSeekClient
    participant DB as Database
    
    loop For each question
        DP->>AV: validate_answer()
        
        alt Reference answer exists
            AV->>DB: Get reference answer
        else No reference answer
            AV->>AV: Use previous correct answers
        end
        
        AV->>DS: check_answer_correctness()
        
        alt DeepSeek API Success
            DS->>AV: Return validation results
        else DeepSeek API Failure
            AV->>AV: _basic_similarity() (fallback)
        end
        
        AV->>DP: Return validation results
        DP->>DB: Update ExtractedQuestion with results
    end
```

## Learning Path Update Process

```mermaid
sequenceDiagram
    participant DP as DocumentProcessor
    participant Celery as Celery Worker
    participant KT as Knowledge Tracing
    participant LP as Learning Path Generator
    participant DB as Database
    
    DP->>DP: _update_learning_paths()
    
    alt Student and Course specified
        DP->>DB: Create PredictionBatch
        DP->>Celery: generate_mastery_predictions.delay()
        
        DP->>DB: Create PathGenerationJob
        DP->>Celery: generate_learning_path.delay()
        
        Celery->>KT: Run knowledge tracing model
        KT->>DB: Update mastery predictions
        
        Celery->>LP: Generate learning path
        LP->>DB: Save learning path recommendations
    end
```

## DeepSeek API Integration Points

### 1. Question Extraction

```python
# In DocumentProcessor._extract_qa_from_text()
prompt = f"""
Extract all questions and their corresponding student answers from the following text.
Format each question-answer pair as:
QUESTION: [question text]
ANSWER: [student's answer]

Text to process:
{text}
"""

messages = [
    {"role": "system", "content": "You are an expert at identifying questions and answers in educational documents."},
    {"role": "user", "content": prompt}
]

# Call DeepSeek API
response = self.deepseek_client._call_api(messages)
content = response['choices'][0]['message']['content']

# Parse the response to extract Q&A pairs
```

### 2. Topic Classification

```python
# In DeepSeekClassifier.classify_question()
# Format topics for the prompt
topic_text = "\n".join([
    f"ID: {topic['id']}, Name: {topic['name']}, Description: {topic.get('description', '')}"
    for topic in available_topics
])

messages = [
    {"role": "system", "content": "You are an expert in Java programming and educational content classification."},
    {"role": "user", "content": f"""
Classify the following Java programming question into the most appropriate topic.

Available topics:
{topic_text}

Question to classify:
{question_text}

Respond with a JSON object with the following structure:
{{
    "topic_id": "The ID of the most relevant topic",
    "confidence": "A number between 0 and 1 indicating your confidence",
    "explanation": "Brief explanation of why this topic is the best match"
}}
"""
    }
]

# Call DeepSeek API
response = self.client._call_api(messages)
```

### 3. Answer Validation

```python
# In AnswerValidator.validate_answer()
messages = [
    {"role": "system", "content": "You are an expert in evaluating Java programming answers."},
    {"role": "user", "content": f"""
Evaluate whether the student's answer to the following Java programming question is correct.

Question:
{question_text}

Correct Answer:
{correct_answer}

Student's Answer:
{student_answer}

Respond with a JSON object with the following structure:
{{
    "is_correct": true/false, # Whether the answer is fundamentally correct
    "score": a number between 0 and 1 indicating the quality of the answer,
    "feedback": "Detailed feedback on the student's answer",
    "explanation": "Explanation of the score and assessment"
}}
"""
    }
]

# Call DeepSeek API
response = self.client._call_api(messages)
