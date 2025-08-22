# PAL 2.0 - Personalized Adaptive Learning System

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![Django](https://img.shields.io/badge/Django-5.1.7-green.svg)](https://djangoproject.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.2-red.svg)](https://pytorch.org)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15-blue.svg)](https://postgresql.org)
[![Docker](https://img.shields.io/badge/Docker-Compose-blue.svg)](https://docker.com)

## 🎯 Overview

**PAL 2.0** is an advanced **Personalized Adaptive Learning System** that leverages machine learning and educational data mining to create customized learning experiences for students. The system uses sophisticated knowledge tracing models to predict student mastery levels and generates personalized learning paths with targeted resource recommendations.

### Key Features

- 🧠 **AI-Powered Knowledge Tracing** - Deep Learning models (DKT, SAKT) for predicting student knowledge states
- 📚 **Personalized Learning Paths** - Adaptive curriculum sequencing based on individual performance
- 🎯 **Intelligent Resource Recommendation** - Context-aware learning material suggestions
- 📊 **Real-time Analytics** - Comprehensive dashboards for students and educators
- 🔄 **Automated Content Processing** - Document ingestion and question extraction
- 🌐 **RESTful API** - Complete API for system integration and mobile apps
- 🐳 **Production Ready** - Fully containerized with Docker Compose

## 🏗️ System Architecture

PAL 2.0 follows a modular Django architecture with specialized applications:

```
PAL 2.0 System
├── Core Learning Engine
│   ├── Student Profiles & Performance Tracking
│   ├── Course & Topic Management
│   └── Assessment & Interaction Recording
├── Machine Learning Pipeline
│   ├── Knowledge Tracing Models (DKT/SAKT)
│   ├── Mastery Prediction Engine
│   └── Model Training & Evaluation
├── Adaptive Learning System
│   ├── Personalized Path Generation
│   ├── Resource Recommendation Engine
│   └── Progress Monitoring
├── Knowledge Graph
│   ├── Topic Relationship Mapping
│   ├── Prerequisite Management
│   └── Curriculum Structure
├── Data Processing Pipeline
│   ├── Document Ingestion (PDF/DOCX)
│   ├── Content Classification
│   └── Data Validation & Quality Control
└── API & Integration Layer
    ├── RESTful API Endpoints
    ├── Authentication & Authorization
    └── Real-time Monitoring
```

## 🛠️ Technology Stack

### Backend Framework
- **Django 5.1.7** - Main web framework with modern features
- **Django REST Framework 3.15.0** - API development and serialization
- **PostgreSQL 15** - Primary database with advanced indexing
- **Redis 7** - Caching, session management, and message broker

### Machine Learning & AI
- **PyTorch 2.1.2** - Deep learning framework for knowledge tracing
- **Transformers 4.40.0+** - NLP models for content analysis
- **Sentence Transformers 3.0.0+** - Text embeddings and similarity
- **Scikit-learn 1.4.0** - Traditional ML algorithms and metrics
- **NetworkX 3.3.0** - Knowledge graph operations and analysis

### Data Processing
- **Pandas 2.2.0** - Data manipulation and analysis
- **NumPy 1.26.3** - Numerical computing and array operations
- **PDFPlumber 0.5.28+** - PDF document processing
- **Python-docx 1.1.2** - Word document processing
- **OpenPyXL 3.1.2** - Excel file handling

### Async Processing & Deployment
- **Celery 5.3.6** - Distributed task queue for ML training
- **Docker & Docker Compose** - Containerization and orchestration
- **Gunicorn 21.2.0** - WSGI server for production
- **Nginx** - Reverse proxy and static file serving (recommended)

## 📁 Project Structure

```
PAL_2.0/
├── 📁 api/                          # REST API endpoints
│   ├── views.py                     # API views and endpoints
│   ├── serializers.py               # Data serialization
│   └── urls.py                      # API URL routing
├── 📁 core/                         # Core learning system
│   ├── models.py                    # Student, Course, Topic, Assessment models
│   ├── views.py                     # Web interface views
│   ├── admin.py                     # Django admin configuration
│   ├── 📁 management/commands/      # Data loading commands
│   │   ├── load_resources.py        # Load learning resources
│   │   ├── load_questions_interactions.py  # Load Q&A data
│   │   └── populate_users_courses.py       # Setup users and courses
│   ├── 📁 services/                 # Business logic services
│   │   └── answer_validation.py     # Answer validation service
│   └── 📁 templates/                # HTML templates
├── 📁 ml_models/                    # Machine Learning pipeline
│   ├── models.py                    # ML model tracking and jobs
│   ├── tasks.py                     # Celery tasks for ML operations
│   ├── 📁 ml/                       # ML implementations
│   │   ├── dkt.py                   # Deep Knowledge Tracing model
│   │   ├── sakt.py                  # Self-Attentive Knowledge Tracing
│   │   └── django_data_preparation.py  # Data preprocessing
│   └── 📁 management/commands/      # ML training commands
│       ├── train_models.py          # Train knowledge tracing models
│       └── train_integrated_models.py  # Integrated training pipeline
├── 📁 learning_paths/               # Adaptive learning paths
│   ├── models.py                    # Learning path and recommendation models
│   ├── tasks.py                     # Path generation tasks
│   ├── 📁 ml/                       # Path generation algorithms
│   │   └── adaptive_path_lstm.py    # LSTM-based path generation
│   ├── 📁 management/commands/      # Path management commands
│   │   └── generate_learning_paths.py  # Generate personalized paths
│   └── 📁 templates/                # Path visualization templates
├── 📁 knowledge_graph/              # Knowledge structure management
│   ├── models.py                    # Knowledge graph and relationships
│   ├── 📁 services/                 # Graph operations
│   │   ├── graph_operations.py      # Graph algorithms and queries
│   │   └── graph_visualizer.py      # Graph visualization utilities
│   └── 📁 management/commands/      # Graph management
│       └── load_ksg.py              # Load knowledge structure graph
├── 📁 data_ingestion/               # Document processing pipeline
│   ├── models.py                    # Upload tracking and extracted content
│   ├── tasks.py                     # Document processing tasks
│   ├── 📁 services/                 # Processing services
│   │   ├── document_processor.py    # PDF/DOCX processing
│   │   ├── csv_processor.py         # CSV data processing
│   │   ├── data_validator.py        # Data validation and cleaning
│   │   └── error_reporter.py        # Error handling and reporting
│   └── 📁 templates/                # Upload interface templates
├── 📁 topic_classification/         # Automated content classification
│   ├── models.py                    # Classification models and jobs
│   ├── tasks.py                     # Classification tasks
│   └── 📁 services/                 # Classification services
│       ├── classifier.py            # Base classifier interface
│       ├── classifier_factory.py    # Classifier factory pattern
│       └── dissect_classifier.py    # Dissect API integration
├── 📁 static/                       # Static files and sample data
│   └── 📁 data/                     # Sample educational content
│       ├── java_resources.json      # 159 Java learning resources
│       ├── java_topics.json         # Java topic hierarchy
│       ├── java_questions.yaml      # Sample questions and assessments
│       ├── java_assessments.json    # Assessment definitions
│       └── java_ksg.json            # Knowledge structure graph
├── 📁 pal_project/                  # Django project configuration
│   ├── settings.py                  # Comprehensive Django settings
│   ├── urls.py                      # Main URL configuration
│   ├── celery.py                    # Celery configuration
│   └── wsgi.py                      # WSGI application
├── 📁 trained_models/               # ML model storage
│   ├── dkt_metadata.json            # DKT model metadata
│   └── metadata.json                # General model metadata
├── 📁 Testing_Data/                 # Sample test documents
│   ├── CS206- Sample Exam.pdf       # Sample PDF exam
│   └── CS206- Sample Exam-2.docx    # Sample DOCX exam
├── 📁 Project_Architecture/         # System architecture diagrams
├── docker-compose.yml               # Multi-service Docker setup
├── Dockerfile                       # Application container definition
├── requirements.txt                 # Python dependencies
├── manage.py                        # Django management script
└── README.md                        # This file
```

## 🧠 Machine Learning Models

### Knowledge Tracing Models

#### 1. Deep Knowledge Tracing (DKT)
- **Architecture**: LSTM-based neural network
- **Purpose**: Predicts student knowledge states over time
- **Input**: Sequence of (topic, correctness) pairs
- **Output**: Probability of correct answer for each topic
- **Features**: 
  - Handles variable-length sequences
  - Captures temporal learning patterns
  - Supports multiple topics simultaneously

#### 2. Self-Attentive Knowledge Tracing (SAKT)
- **Architecture**: Transformer-based with self-attention
- **Purpose**: Advanced knowledge state modeling with attention mechanisms
- **Input**: Exercise sequences with embeddings
- **Output**: Knowledge state predictions with attention weights
- **Features**:
  - Attention-based learning pattern recognition
  - Better handling of long sequences
  - Interpretable attention weights

### Model Training Pipeline

```python
# Example: Training a DKT model
python manage.py train_models --model-type dkt --course CS101 --epochs 50

# Example: Generating predictions
python manage.py train_integrated_models --course CS101 --batch-size 32
```

## 📊 Data Models

### Core Educational Entities

#### Student Profile
```python
class Student(models.Model):
    student_id = models.CharField(max_length=50, primary_key=True)
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    major = models.CharField(max_length=100)
    academic_level = models.CharField(max_length=20)  # freshman, sophomore, etc.
    gpa = models.FloatField()
    prior_knowledge_score = models.FloatField()
    study_frequency = models.CharField(max_length=20)
    attendance_rate = models.FloatField()
    participation_score = models.FloatField()
    # ... additional fields for comprehensive profiling
```

#### Learning Interaction Tracking
```python
class StudentInteraction(models.Model):
    student = models.ForeignKey(Student, on_delete=models.CASCADE)
    question = models.ForeignKey(Question, on_delete=models.CASCADE)
    response = models.TextField()
    correct = models.BooleanField()
    score = models.FloatField()
    time_taken = models.DurationField()
    timestamp = models.DateTimeField()
    attempt_number = models.PositiveIntegerField()
    resource_viewed_before = models.BooleanField()
```

### Learning Path System

#### Personalized Learning Path
```python
class LearningPath(models.Model):
    student = models.ForeignKey(Student, on_delete=models.CASCADE)
    course = models.ForeignKey(Course, on_delete=models.CASCADE)
    name = models.CharField(max_length=200)
    generated_at = models.DateTimeField(auto_now_add=True)
    status = models.CharField(max_length=20)  # active, completed, archived
    student_stats = models.JSONField()  # Flexible student profiling
    total_estimated_time = models.FloatField()
    overall_progress = models.FloatField()
```

#### Resource Recommendations
```python
class TopicResource(models.Model):
    weak_topic = models.ForeignKey(WeakTopic, on_delete=models.CASCADE)
    recommended_topic = models.ForeignKey(RecommendedTopic, on_delete=models.CASCADE)
    title = models.CharField(max_length=300)
    url = models.URLField(max_length=500)
    resource_type = models.CharField(max_length=50)  # video, documentation, etc.
    difficulty = models.CharField(max_length=20)
    estimated_time = models.FloatField()
    # Progress tracking fields
    viewed = models.BooleanField(default=False)
    completed = models.BooleanField(default=False)
    rating = models.IntegerField(null=True, blank=True)
```

## 🚀 Installation & Setup

### Prerequisites

- **Python 3.9+**
- **Docker & Docker Compose**
- **Git**
- **PostgreSQL 15** (if running without Docker)
- **Redis 7** (if running without Docker)

### Quick Start with Docker (Recommended)

1. **Clone the repository**
```bash
git clone https://github.com/bkshgtm/PersonalizedAdaptiveLearning.git
cd PAL_2.0
```

2. **Create environment file**
```bash
cp .env.example .env
# Edit .env with your configuration
```

3. **Build and start services**
```bash
docker-compose up --build
```

4. **Initialize the database**
```bash
docker-compose exec web python manage.py migrate
docker-compose exec web python manage.py createsuperuser
```

5. **Load sample data**
```bash
# Load Java programming curriculum
docker-compose exec web python manage.py populate_users_courses
docker-compose exec web python manage.py load_resources
docker-compose exec web python manage.py load_questions_interactions
docker-compose exec web python manage.py load_ksg
```

6. **Access the application**
- **Web Interface**: http://localhost:8000
- **Admin Panel**: http://localhost:8000/admin
- **API Documentation**: http://localhost:8000/api/docs/

### Manual Installation

1. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Setup PostgreSQL database**
```bash
createdb pal_db
```

4. **Configure environment variables**
```bash
export DATABASE_URL="postgresql://username:password@localhost:5432/pal_db"
export SECRET_KEY="your-secret-key"
export DEBUG=True
export REDIS_URL="redis://localhost:6379/0"
```

5. **Run migrations and setup**
```bash
python manage.py migrate
python manage.py createsuperuser
python manage.py collectstatic
```

6. **Start services**
```bash
# Terminal 1: Django development server
python manage.py runserver

# Terminal 2: Celery worker
celery -A pal_project worker -l info

# Terminal 3: Celery beat scheduler
celery -A pal_project beat -l info
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the project root:

```env
# Django Configuration
SECRET_KEY=your-super-secret-key-here
DEBUG=False
ALLOWED_HOSTS=localhost,127.0.0.1,your-domain.com

# Database Configuration
DATABASE_URL=postgresql://pal_user:pal_password@db:5432/pal_db

# Redis Configuration
REDIS_URL=redis://redis:6379/0
CELERY_BROKER_URL=redis://redis:6379/0
CELERY_RESULT_BACKEND=django-db

# ML Model Configuration
MODEL_STORAGE_PATH=/app/models
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# CORS Configuration
CORS_ALLOWED_ORIGINS=http://localhost:3000,http://localhost:8000

# API Keys (Optional)
OPENAI_API_KEY=your-openai-key
ANTHROPIC_API_KEY=your-anthropic-key
```

### Docker Services

The `docker-compose.yml` defines these services:

- **web**: Django application server
- **db**: PostgreSQL database
- **redis**: Redis cache and message broker
- **celery**: Background task worker
- **celery-beat**: Periodic task scheduler

## 📚 Usage Examples

### 1. Training Knowledge Tracing Models

```bash
# Train DKT model for a specific course
python manage.py train_models --model-type dkt --course CS101 --epochs 50 --batch-size 32

# Train SAKT model with custom parameters
python manage.py train_models --model-type sakt --course CS101 --hidden-size 128 --num-heads 8

# Train integrated models (recommended)
python manage.py train_integrated_models --course CS101
```

### 2. Generating Learning Paths

```bash
# Generate learning paths for all students in a course
python manage.py generate_learning_paths --course CS101

# Generate path for specific student
python manage.py generate_learning_paths --student STU001 --course CS101

# Test the complete system
python manage.py test_complete_system --course CS101 --num-students 10
```

### 3. Data Loading and Management

```bash
# Load educational resources
python manage.py load_resources --file static/data/java_resources.json

# Load questions and interactions
python manage.py load_questions_interactions --file static/data/java_questions.yaml

# Load knowledge structure graph
python manage.py load_ksg --file static/data/java_ksg.json

# Populate sample users and courses
python manage.py populate_users_courses
```

### 4. API Usage Examples

#### Authentication
```bash
# Get authentication token
curl -X POST http://localhost:8000/api/auth/token/ \
  -H "Content-Type: application/json" \
  -d '{"username": "your_username", "password": "your_password"}'
```

#### Student Profile and Learning Path
```bash
# Get student profile with knowledge states
curl -X GET "http://localhost:8000/api/students/STU001/profile/?course_id=CS101" \
  -H "Authorization: Token your-token-here"

# Generate recommendations for a student
curl -X POST http://localhost:8000/api/students/STU001/recommendations/ \
  -H "Authorization: Token your-token-here" \
  -H "Content-Type: application/json" \
  -d '{"course_id": "CS101"}'
```

#### System Monitoring
```bash
# Get system dashboard data
curl -X GET http://localhost:8000/api/monitoring/dashboard/ \
  -H "Authorization: Token your-token-here"

# Get course-specific analytics
curl -X GET http://localhost:8000/api/courses/CS101/recommendations/ \
  -H "Authorization: Token your-token-here"
```

## 🔌 API Reference

### Authentication Endpoints
- `POST /api/auth/token/` - Obtain authentication token
- `POST /api/auth/refresh/` - Refresh authentication token

### Student Management
- `GET /api/students/{student_id}/profile/` - Get student profile and knowledge states
- `POST /api/students/{student_id}/recommendations/` - Generate learning recommendations
- `GET /api/students/{student_id}/path/` - Get active learning path

### Learning Analytics
- `GET /api/courses/{course_id}/recommendations/` - Course-wide analytics
- `GET /api/monitoring/dashboard/` - System monitoring dashboard
- `GET /api/recommendation-status/` - Check recommendation generation status

### Data Management
- `POST /api/data-ingestion/upload/` - Upload CSV data
- `GET /api/knowledge-graph/active/` - Get active knowledge graph
- `GET /api/ml-models/models/` - List available ML models

### Content Validation
- `POST /api/validate-answer/` - Validate student answers using AI

## 📈 Sample Data

The system includes comprehensive sample data for Java programming education:

### Java Programming Curriculum
- **20 Topics**: From basic syntax to advanced GUI programming
- **159 Learning Resources**: Videos, documentation, tutorials, and interactive tools
- **Multiple Resource Types**: Video, documentation, visual, practice, other
- **Difficulty Levels**: Beginner, intermediate, advanced
- **Estimated Time**: Detailed time estimates for each resource

### Topic Hierarchy
1. **Java Program Structure and Compilation** (25 min)
2. **Comments and Print Statements** (15 min)
3. **Variables and Data Types** (35 min)
4. **Arithmetic and Assignment Operators** (22 min)
5. **Relational and Logical Operators** (20 min)
6. **Conditional Statements** (30 min)
7. **While and Do-While Loops** (25 min)
8. **For Loops and Nested Loops** (30 min)
9. **Methods and Functions** (40 min)
10. **Method Parameters and Return Values** (32 min)
11. **Arrays and 2D Arrays** (45 min)
12. **ArrayList and Collections** (35 min)
13. **Classes and Objects** (40 min)
14. **Constructors and Object Initialization** (35 min)
15. **Encapsulation and Access Modifiers** (32 min)
16. **Inheritance and Polymorphism** (45 min)
17. **Interfaces and Abstract Classes** (40 min)
18. **Exception Handling** (45 min)
19. **File I/O Operations** (50 min)
20. **JavaFX and GUI Programming** (1 hour)

## 🧪 Testing

### Running Tests

```bash
# Run all tests
python manage.py test

# Run specific app tests
python manage.py test core
python manage.py test ml_models
python manage.py test learning_paths

# Run with coverage
pip install coverage
coverage run --source='.' manage.py test
coverage report
coverage html
```

### Test Data Generation

```bash
# Generate test interactions
python manage.py test_complete_system --course CS101 --num-students 50

# Validate system integrity
python manage.py check_ksg_data
```

## 🚀 Deployment

### Production Deployment with Docker

1. **Update environment for production**
```env
DEBUG=False
ALLOWED_HOSTS=your-domain.com,www.your-domain.com
DATABASE_URL=postgresql://user:pass@prod-db:5432/pal_db
```

2. **Use production Docker Compose**
```bash
docker-compose -f docker-compose.prod.yml up -d
```

3. **Setup reverse proxy (Nginx)**
```nginx
server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
    
    location /static/ {
        alias /path/to/static/files/;
    }
}
```

### Scaling Considerations

- **Database**: Use PostgreSQL with connection pooling
- **Cache**: Redis cluster for high availability
- **Workers**: Scale Celery workers based on ML training load
- **Storage**: Use cloud storage for model files and uploads
- **Monitoring**: Implement logging and monitoring (Sentry, DataDog)

## 🤝 Contributing

### Development Workflow

1. **Fork the repository**
2. **Create feature branch**
```bash
git checkout -b feature/amazing-feature
```

3. **Make changes and test**
```bash
python manage.py test
black .
isort .
flake8 .
```

4. **Commit and push**
```bash
git commit -m "Add amazing feature"
git push origin feature/amazing-feature
```

5. **Create Pull Request**

### Code Style

- **Python**: Follow PEP 8, use Black formatter
- **Django**: Follow Django best practices
- **ML Code**: Document model architectures and hyperparameters
- **API**: Follow RESTful conventions
- **Tests**: Maintain >80% code coverage

### Development Tools

```bash
# Install development dependencies
pip install black isort flake8 pytest-django coverage

# Format code
black .
isort .

# Lint code
flake8 .

# Type checking (optional)
pip install mypy
mypy .
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Django Community** - For the excellent web framework
- **PyTorch Team** - For the deep learning framework
- **Educational Data Mining Community** - For research insights
- **Open Source Contributors** - For various libraries and tools

## 📞 Support

- **Documentation**: [Project Wiki](https://github.com/bkshgtm/PersonalizedAdaptiveLearning/wiki)
- **Issues**: [GitHub Issues](https://github.com/bkshgtm/PersonalizedAdaptiveLearning/issues)
- **Discussions**: [GitHub Discussions](https://github.com/bkshgtm/PersonalizedAdaptiveLearning/discussions)
- **Email**: [Contact the maintainers](mailto:support@pal-system.com)

## 🔮 Roadmap

### Version 2.1 (Q2 2025)
- [ ] Advanced attention mechanisms in SAKT
- [ ] Multi-modal learning resource support
- [ ] Real-time collaborative learning features
- [ ] Mobile application development

### Version 2.2 (Q3 2025)
- [ ] Federated learning for privacy-preserving training
- [ ] Advanced natural language processing for content analysis
- [ ] Gamification and engagement features
- [ ] Integration with popular LMS platforms

### Version 3.0 (Q4 2025)
- [ ] Multi-language support
- [ ] Advanced visualization and learning analytics
- [ ] AI-powered content generation
- [ ] Blockchain-based credential verification

---

**PAL 2.0** - Transforming education through personalized, adaptive learning powered by artificial intelligence.

*Built with ❤️ by the PAL Development Team*
