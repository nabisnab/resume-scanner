# Resume Scanner - NLP Powered Resume Analysis Tool

An intelligent resume scanning application powered by NLP that analyzes resumes and scores them against job descriptions.

## Features

🧠 **Advanced NLP Processing**
- Named Entity Recognition (NER) for skill extraction
- Semantic similarity analysis using Sentence-BERT
- Text processing with spaCy and NLTK

📊 **Comprehensive Scoring**
- Multi-category scoring system
- Detailed feedback and recommendations
- ATS optimization suggestions

🎨 **User-Friendly Interface**
- Modern, responsive design
- Drag-and-drop resume upload
- Real-time analysis and results

## Tech Stack

**Backend:**
- FastAPI
- spaCy
- NLTK
- Sentence-BERT
- SQLAlchemy

**Frontend:**
- HTML5
- CSS3
- Vanilla JavaScript

## Project Structure

```
resume-scanner/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py
│   │   ├── config.py
│   │   ├── models/
│   │   │   └── schemas.py
│   │   ├── api/
│   │   │   ├── user_routes.py
│   │   │   └── admin_routes.py
│   │   ├── services/
│   │   │   ├── nlp_processor.py
│   │   │   ├── scoring_engine.py
│   │   │   └── recommendation_engine.py
│   │   ├── utils/
│   │   │   └── file_handler.py
│   │   └── db/
│   │       └── database.py
│   ├── requirements.txt
│   └── .env.example
├── frontend/
│   ├── index.html
│   ├── upload.html
│   ├── results.html
│   ├── css/
│   │   └── style.css
│   └── js/
│       ├── main.js
│       └── api.js
└── README.md
```

## Getting Started

### Backend Setup

1. Navigate to the backend directory:
```bash
cd backend
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create .env file:
```bash
cp .env.example .env
```

5. Run the server:
```bash
uvicorn app.main:app --reload
```

### Frontend Setup

1. Navigate to the frontend directory:
```bash
cd frontend
```

2. Open `index.html` in your browser or use a local server:
```bash
python -m http.server 8000
```

3. Visit `http://localhost:8000`

## API Endpoints

### User APIs
- `POST /user/upload-resume` - Upload resume and job position
- `POST /user/get-score` - Get scoring results
- `GET /user/recommendations` - Get recommendations

### Admin APIs
- `GET /admin/all-resumes` - View all resumes
- `GET /admin/rankings` - View rankings
- `POST /admin/update-weights` - Update weights

## License

MIT