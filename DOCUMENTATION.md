# Text Summarization using LSA and T5 Transformer - Documentation

## 📋 Table of Contents
1. [Project Overview](#project-overview)
2. [Technology Stack](#technology-stack)
3. [Architecture](#architecture)
4. [Project Structure](#project-structure)
5. [Installation & Setup](#installation--setup)
6. [Core Features](#core-features)
7. [Algorithms](#algorithms)
8. [Database Schema](#database-schema)
9. [API Endpoints](#api-endpoints)
10. [User Guide](#user-guide)
11. [Development Guide](#development-guide)
12. [Troubleshooting](#troubleshooting)

---

## 🎯 Project Overview

This is a **Bachelor's final year project** on text summarization that combines two powerful approaches:
- **Latent Semantic Analysis (LSA)**: Classical unsupervised machine learning technique
- **T5 Transformer**: Modern transformer-based pre-trained model

The project provides a web-based interface for users to summarize text using either algorithm, with detailed step-by-step visualization of the LSA process.

### Key Features
✅ Multi-algorithm summarization (LSA and T5)  
✅ User authentication and history tracking  
✅ Real-time visualization of LSA intermediate steps  
✅ Topic extraction from summaries  
✅ Adjustable summary length  
✅ Comprehensive summary details (word count, sentence count)  
✅ User dashboard for managing summarization history  

---

## 🛠️ Technology Stack

### Backend
- **Framework**: Django 5.0
- **Language**: Python 3.x
- **ML Libraries**: 
  - `transformers==4.36.2` (T5 model)
  - `torch==2.1.2` & `torchvision==0.16.2` (PyTorch)
  - `numpy==1.26.2` (Numerical computations)
  - `nltk==3.8.1` (Natural Language Processing)
  - `scikit-learn` (Machine Learning utilities)

### Frontend
- **Languages**: HTML, CSS, JavaScript
- **Framework**: Django Templates
- **Communication**: AJAX/JSON for asynchronous requests

### Database
- SQLite (default Django ORM)

### Deployment
- Vercel (see `vercel.json`)

### Language Composition
- HTML: 44.4%
- Python: 35.8%
- JavaScript: 17%
- CSS: 2.8%

---

## 🏗️ Architecture

### High-Level Flow

```
┌─────────────────────────────────────────────────────────────┐
│                     User Interface                           │
│              (HTML, CSS, JavaScript)                        │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼ AJAX/JSON
┌─────────────────────────────────────────────────────────────┐
│                   Django Views                              │
│              (Request Handler)                              │
└────────────────────┬────────────────────────────────────────┘
                     │
        ┌────────────┴────────────┐
        │                         │
        ▼                         ▼
┌──────────────────┐    ┌──────────────────┐
│  LSA Summarizer  │    │  T5 Summarizer   │
│  (Class-based)   │    │  (Transformer)   │
└────────┬─────────┘    └────────┬─────────┘
         │                       │
         └───────────┬───────────���
                     │
        ┌────────────┴────────────┐
        │                         │
        ▼                         ▼
┌──────────────────┐    ┌──────────────────┐
│ Topic Extractor  │    │   Database       │
│ (TF-IDF)         │    │   (Django ORM)   │
└──────────────────┘    └──────────────────┘
```

---

## 📁 Project Structure

```
Text-Summarization-using-LSA-and-T5-Transformer/
│
├── summarizer_project/              # Django Project Configuration
│   ├── __init__.py
│   ├── settings.py                  # Django settings
│   ├── urls.py                      # Main URL routing
│   ├── asgi.py                      # ASGI configuration
│   └── wsgi.py                      # WSGI configuration
│
├── summarizer_app/                  # Django Application
│   ├── migrations/                  # Database migrations
│   ├── models/                      # Pre-trained models directory
│   ├── static/                      # Static files (CSS, JS)
│   ├── templates/                   # HTML templates
│   │   ├── front_page.html
│   │   ├── homepage.html
│   │   ├── home.html
│   │   ├── about_us.html
│   │   ├── dashboard.html
│   │   ├── summary_page.html
│   │   └── slides.html
│   │
│   ├── admin.py                     # Django admin configuration
│   ├── apps.py                      # App configuration
│   ├── forms.py                     # Django forms
│   ├── models.py                    # Database models
│   ├── views.py                     # Request handlers (16KB)
│   ├── urls.py                      # App URL routing
│   ├── tests.py                     # Unit tests
│   ├── middleware.py                # Custom middleware
│   │
│   ├── lsa_summarizer.py            # LSA Algorithm Implementation
│   ├── t5_summarizer.py             # T5 Transformer Implementation
│   ├── topic_extractor.py           # TF-IDF Topic Extraction
│   └── lsa_old.md                   # Legacy LSA documentation
│
├── manage.py                        # Django management script
├── requirements.txt                 # Python dependencies (34 packages)
├── README.md                        # Quick start guide
├── guide.md                         # Implementation guide
├── vercel.json                      # Vercel deployment config
└── .gitignore                       # Git ignore rules
```

---

## 📦 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- Virtual environment (recommended)

### Step 1: Clone Repository
```bash
git clone https://github.com/Suman-Khadka-2002/Text-Summarization-using-LSA-and-T5-Transformer.git
cd Text-Summarization-using-LSA-and-T5-Transformer
```

### Step 2: Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

**Key Dependencies:**
- Django 5.0
- Transformers 4.36.2 (T5 model)
- PyTorch 2.1.2
- NumPy 1.26.2
- NLTK 3.8.1

### Step 4: Download T5 Model
The project uses a custom fine-tuned T5 model. Ensure the model is placed at:
```
summarizer_app/models/pubmed_model/
```

If using the base T5 model instead:
```bash
# The model will auto-download on first run
# Requires ~750 MB of disk space
```

### Step 5: Apply Migrations
```bash
python manage.py migrate
```

### Step 6: Create Superuser (Optional)
```bash
python manage.py createsuperuser
```

### Step 7: Run Development Server
```bash
python manage.py runserver
```

Access the application at: `http://127.0.0.1:8000/`

---

## ✨ Core Features

### 1. User Authentication
- **Registration**: Create new user accounts with email validation
- **Login/Logout**: Secure session-based authentication
- **User Profiles**: Track user-specific summarization history

### 2. Text Summarization
**Supported Algorithms:**

#### LSA (Latent Semantic Analysis)
- Classical unsupervised learning approach
- Shows all intermediate steps (eigen decomposition, scoring)
- Best for short to medium documents
- Deterministic results

#### T5 Transformer
- Modern deep learning approach
- Uses pre-trained transformer model
- Best for understanding semantic meaning
- Variable results (stochastic)

#### T5 Fine-tuned Model
- Custom trained model on specific domain
- Located at: `summarizer_app/models/pubmed_model/`

### 3. Summary Customization
- **Length Control**: Adjust summary length via range slider
- **Algorithm Selection**: Choose between LSA, T5, or fine-tuned T5
- **Real-time Processing**: Immediate summary generation

### 4. Topic Extraction
- Automatically extracts key topics using TF-IDF
- Provides short topic summary
- Helps understand summary content quickly

### 5. History Management
- View all past summaries in dashboard
- Delete individual history items
- Clear entire history
- Export summary details

### 6. Analytics & Visualization
For LSA summaries, users can view:
- Clean text preprocessing steps
- Normalized text
- Tokenization results
- Term-document matrix
- TF-IDF matrix
- Covariance matrix
- Eigenvalues and eigenvectors
- Sentence scores
- Selected summary sentences

---

## 🧠 Algorithms

### LSA (Latent Semantic Analysis)

**File**: `summarizer_app/lsa_summarizer.py`

**Process:**
```
1. Text Cleaning
   └─ Remove special characters
   └─ Keep alphanumeric and punctuation
   └─ Remove extra whitespace

2. Text Normalization
   └─ Convert to lowercase

3. Sentence Tokenization
   └─ Split on .!? punctuation

4. Word Tokenization
   └─ Split sentences into words

5. Create Term-Document Matrix
   └─ Rows: words, Columns: sentences
   └─ Values: word frequency

6. Calculate Document-Term Matrix (DTM)
   └─ Transpose of term-document matrix

7. Mean Centering
   └─ Subtract column means

8. Covariance Matrix
   └─ Compute using NumPy: cov(DTM.T)

9. Eigenvalue Decomposition
   └─ Find eigenvalues and eigenvectors
   └─ Sort in descending order

10. Sentence Scoring
    └─ Score based on reduced components
    └─ Weighted by eigenvalues

11. Select Top Sentences
    └─ Choose N sentences with highest scores
    └─ Maintain original order

12. Format Summary
    └─ Capitalize and join with periods
```

**Mathematical Foundation:**
```
Covariance Matrix = (DTM.T * DTM) / (n-1)
Sentence Score = | Sum(concept_scores * eigenvalues) |
```

**Complexity:**
- Time: O(m*n + n³) where m=words, n=sentences
- Space: O(m*n)

### T5 Transformer

**File**: `summarizer_app/t5_summarizer.py`

**Model**: `t5-small` or custom fine-tuned variant

**Process:**
```
1. Text Preprocessing
   └─ Clean special characters
   └─ Remove digits
   └─ Normalize whitespace

2. Tokenization
   └─ Encode input with "summarize: " prefix
   └─ Max length: 512 tokens
   └─ Truncate if needed

3. Model Inference
   └─ T5ForConditionalGeneration
   └─ Beam search: 4 beams
   └─ Length penalty: 2.0
   └─ Early stopping: enabled

4. Decoding
   └─ Decode token IDs to text
   └─ Remove special tokens

5. Post-processing
   └─ Capitalize first letter
   └─ Add period at end
   └─ Remove leading colons
```

**Parameters:**
- `max_length`: 512 (input tokens)
- `summary_length`: User-defined output length
- `length_penalty`: 2.0 (encourages longer outputs)
- `num_beams`: 4 (beam width)
- `early_stopping`: True

---

## 💾 Database Schema

### Summary Model
```python
class Summary(models.Model):
    user              # ForeignKey(User) - Owner of summary
    input_text        # TextField - Original text (unlimited)
    algorithm         # CharField(50) - "LSA", "T5", or "T5_our"
    summary_length    # IntegerField - Requested summary length
    summarized_text   # TextField - Generated summary
    short_topic       # TextField - Extracted topics
    created_at        # DateTimeField - Auto timestamp
    
    # Foreign Key Relationship
    user → Django User model (CASCADE delete)
```

**Indexes:**
- Primary: `id`
- Secondary: `user_id` (for user-specific queries)

**Example Query:**
```python
# Get all summaries by user
summaries = Summary.objects.filter(user=request.user)

# Get specific summary
summary = Summary.objects.get(id=summary_id)

# Delete summary
summary.delete()
```

---

## 🔌 API Endpoints

### Authentication
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/register/` | POST | Register new user |
| `/login/` | POST | Authenticate user |
| `/logout/` | POST | End user session |

### Summarization
| Endpoint | Method | Parameters | Returns |
|----------|--------|------------|---------|
| `/summarize/` | GET | - | User summary history (JSON) |
| `/summarize/` | POST | input_text, algorithm, summary_length, generated_summary | Summary details (JSON) |

**POST Parameters:**
```json
{
  "input_text": "Your text here...",
  "algorithm": "LSA|T5|T5_our",
  "summary_length": 50,
  "generated_summary": "Pre-generated summary (optional)"
}
```

**Response:**
```json
{
  "generated_summary": "Summary text...",
  "word_count": 45,
  "sentence_count": 3,
  "short_topic": "extracted topics",
  
  // For LSA only:
  "clean_text": "...",
  "normalized_text": "...",
  "tokenized_sentences": [...],
  "term_document_matrix": {...},
  "covariance_matrix": [...],
  "eigenvalues": [...],
  "eigenvectors": [...],
  "sentence_scores": [...]
}
```

### History Management
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/delete-history/<history_id>/` | DELETE | Delete single history item |
| `/delete-all-history/` | DELETE | Clear entire history |

### Analytics
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/fetch-lsa-summary-details/` | GET | Get LSA intermediate results |
| `/fetch-tfidf-details/` | GET | Get TF-IDF analysis results |

### Dashboard
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/dashboard/` | GET | User summary history page |
| `/summary/<summary_id>/` | GET | View summary details |

---

## 👥 User Guide

### Getting Started

#### 1. Registration
```
1. Click "Register" on homepage
2. Enter: First Name, Last Name, Username, Email
3. Create Password (min 8 characters recommended)
4. Confirm Password
5. Click "Register"
```

#### 2. Login
```
1. Click "Login" on homepage
2. Enter Username and Password
3. Click "Login"
4. Redirected to Home page
```

#### 3. Summarize Text

**Using LSA:**
```
1. Paste or type text in "Original Text" area
2. Select "LSA" from algorithm dropdown
3. Adjust "Summary Length" (1-10 sentences)
4. Click "Summarize" button
5. View results with step-by-step breakdown
```

**Using T5:**
```
1. Paste or type text in "Original Text" area
2. Select "T5" or "T5_our" from dropdown
3. Set summary length
4. Click "Summarize"
5. View generated summary
```

#### 4. View History
```
1. Go to Dashboard
2. See all past summaries
3. Click summary to view full details
4. Delete individual items or clear all
```

#### 5. Account Management
```
1. Dashboard → Settings
2. View profile information
3. Change password
4. Delete account (irreversible)
```

### Tips & Best Practices

**For LSA:**
- ✅ Works best with well-structured documents (500+ words)
- ✅ Good for news articles, research papers
- ✅ Set summary length to 10-20% of original
- ⚠️ Sensitive to text quality

**For T5:**
- ✅ Works with shorter texts (100+ words)
- ✅ Better semantic understanding
- ✅ Less sensitive to formatting
- ⚠️ Slower on CPU-only systems

**Text Preparation:**
- Remove unnecessary whitespace
- Ensure proper sentence structure
- Avoid broken paragraphs
- Clean special characters if needed

---

## 👨‍💻 Development Guide

### Adding New Summarization Algorithm

#### 1. Create Algorithm Class
```python
# summarizer_app/my_algorithm.py
class MyAlgorithmSummarizer:
    def __init__(self):
        # Initialize model/tokenizer
        pass
    
    def preprocess_text(self, text):
        # Clean and prepare text
        return cleaned_text
    
    def summarize(self, text, length):
        # Implement summarization logic
        return summary
```

#### 2. Update Views
```python
# summarizer_app/views.py
from .my_algorithm import MyAlgorithmSummarizer

algorithms = {
    "T5": t5_summarize,
    "LSA": "LSA",
    "T5_our": "T5_our",
    "MyAlgorithm": "MyAlgorithm",  # Add here
}
```

#### 3. Add Template Option
```html
<!-- templates/homepage.html -->
<select name="algorithm">
    <option value="LSA">LSA</option>
    <option value="T5">T5</option>
    <option value="MyAlgorithm">My Algorithm</option>
</select>
```

### Running Tests
```bash
python manage.py test summarizer_app
```

### Database Migrations
```bash
# Create migration
python manage.py makemigrations

# Apply migration
python manage.py migrate

# Revert migration
python manage.py migrate summarizer_app 0001
```

### Code Style
- Follow PEP 8 guidelines
- Add docstrings to functions
- Use meaningful variable names
- Comment complex logic

### Debugging
```python
# Enable Django debug logging
import logging
logger = logging.getLogger(__name__)
logger.debug("Debug message")

# Print to console in views
print("Value:", variable)
```

---

## 🔧 Troubleshooting

### Common Issues

#### 1. Model Download Fails
**Problem**: T5 model not downloading on first run
```
FileNotFoundError: Can't find model 'summarizer_app/models/pubmed_model'
```

**Solution:**
```bash
# Download model manually
python -c "from transformers import T5ForConditionalGeneration, T5Tokenizer; \
T5ForConditionalGeneration.from_pretrained('t5-small'); \
T5Tokenizer.from_pretrained('t5-small')"
```

#### 2. CUDA/PyTorch Issues
**Problem**: PyTorch not recognizing GPU

**Solution:**
```bash
# Check PyTorch version
python -c "import torch; print(torch.cuda.is_available())"

# Reinstall for CPU only
pip install torch==2.1.2+cpu -f https://download.pytorch.org/whl/torch_stable.html
```

#### 3. Migration Errors
**Problem**: Database migration conflicts

**Solution:**
```bash
# Reset database (development only)
rm db.sqlite3
python manage.py migrate

# Check migration status
python manage.py showmigrations
```

#### 4. CSRF Token Errors
**Problem**: CSRF verification failed

**Solution:**
- Ensure `@csrf_exempt` decorator on API endpoints
- Include CSRF token in AJAX requests
- Check `CSRF_TRUSTED_ORIGINS` in settings.py

#### 5. Memory Issues with Large Text
**Problem**: Out of memory error during summarization

**Solution:**
```python
# Truncate input text
if len(text) > 10000:
    text = text[:10000]

# Reduce batch size in settings
```

#### 6. Slow Summarization
**Problem**: T5 summarization takes too long

**Solution:**
```python
# Use smaller model
"t5-small" instead of "t5-base"

# Reduce beam search
num_beams=2 instead of 4

# Enable GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
```

### Performance Optimization

#### 1. Model Caching
```python
# Load model once, reuse for all requests
from django.views.decorators.cache import cache_page

@cache_page(60 * 5)  # Cache for 5 minutes
def summarize(request):
    pass
```

#### 2. Database Optimization
```python
# Use select_related for foreign keys
summaries = Summary.objects.select_related('user').filter(user=request.user)

# Use only() to limit fields
summaries = Summary.objects.only('id', 'summarized_text').all()
```

#### 3. Frontend Optimization
```javascript
// Debounce input changes
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}
```

---

## 📝 Additional Resources

### Documentation Files
- `README.md` - Quick start guide
- `guide.md` - Implementation guide
- `DOCUMENTATION.md` - This file

### External References
- [Django Documentation](https://docs.djangoproject.com/)
- [Transformers Documentation](https://huggingface.co/docs/transformers/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [NLTK Documentation](https://www.nltk.org/)

### Related Papers
- "Latent Semantic Analysis" - Scott Deerwester et al.
- "T5: Text-to-Text Transfer Transformer" - Colin Raffel et al.

---

