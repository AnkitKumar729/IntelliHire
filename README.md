# IntelliHire - Career Intelligence System

A full-stack, premium web application that uses advanced NLP and machine learning to analyze resumes, extract skills, identify skill gaps, and match candidates with job opportunities.

IntelliHire has been rebuilt with a modern **Xtract Dark Design System** — featuring high-fidelity Framer Motion micro-interactions, staggered layout transitions, glassmorphic UI elements, and highly immersive dashboard aesthetics.

## Features

### For Students/Users
- **Premium Candidate Dashboard**: Navigate an intuitive dark-themed interface with buttery-smooth tab transitions.
- **Resume Upload & OCR Analysis**: Upload digital or scanned PDF resumes. Built-in EasyOCR rasterizes and extracts text from image-based PDFs automatically.
- **AI Resume Feedback**: Get section-by-section AI-powered critique and rewrite suggestions dynamically tailored to specific job descriptions using LLMs (Groq).
- **Dynamic Skills Ontology**: Skills and proficiencies are evaluated against a massive, MongoDB-persisted open skills database containing over 30,000 unique keyless skills.
- **Skill Strength & Gap Analysis**: Get proficiency scores based on experience context and pinpoint exact skills needed for your target jobs.
- **Personalized Learning Path**: Get recommended learning sequences generated through a complex skill-dependency graphing algorithm.
- **Live Job Matching**: Find internships and jobs matched to your profile using deep AI similarity scoring.

### For Recruiters
- **Dynamic Recruiter Dashboard**: High-performance dashboard with animated, staggered statistic count-ups and seamless layout groups.
- **Job Posting & Management**: Create job listings via spring-animated modals with real-time backdrop blur effects.
- **Multi-Model Candidate Ranking**: View AI-matched candidates sorted dynamically using configurable ranking algorithms.
- **Model Comparison Engine**: Automatically fetches and compares candidate rankings across BERT, TF-IDF, and XGBoost simulations, featuring interactive variance bar charts.
- **Recruitment Pipeline**: Track, shortlist, and interact with candidate application statuses instantly via UI pipeline actions.

## Technology Stack

### Backend
- **FastAPI**: Modern, lightning-fast web framework
- **PyMongo**: NoSQL Database management with MongoDB
- **Groq API**: High-speed, advanced LLM integration for resume critique and rewriting
- **EasyOCR / PyMuPDF (fitz)**: Advanced optical character recognition and PDF rendering
- **Sentence-Transformers (BERT)**: Deep learning cosine similarity and embedding generation
- **pdfplumber**: High-speed digital PDF text extraction
- **scikit-learn**: TF-IDF vectorization
- **NetworkX**: Skill dependency graph construction
- **NumPy / Pandas**: Complex data handling

### Frontend
- **React 18 & Vite**: Blazing-fast frontend and tooling
- **Framer Motion**: Enterprise-grade physics, springs, layout transitions, and stagger cascades
- **Tailwind CSS**: Utility-first CSS framework natively integrated into the dark-mode aesthetic
- **shadcn/ui**: Accessible, customizable component architecture
- **Recharts**: Data visualization and analytics charts
- **Axios**: Promised-based HTTP client
- **Lucide React**: Premium icon library

## Project Structure

```
intellihire/
├── backend/
│   ├── main.py              # FastAPI application with advanced NLP pipeline
│   ├── uploads/             # Ephemeral uploaded resume storage
│   └── (MongoDB backend)    # Local or Atlas MongoDB Cluster
├── app/                     # React 18 Frontend
│   ├── src/
│   │   ├── App.jsx          # Main application routing
│   │   ├── App.css          # Core design tokens
│   │   ├── pages/           # Dashboard views (Recruiter, Candidate)
│   │   └── components/      # Reusable motion components (AnimatedStat, FloatingHint, etc)
│   └── package.json
├── start_backend.py         # Backend startup script
└── README.md
```

## Running the Application

### 1. Database Setup
Ensure you have MongoDB running locally on port `27017` or supply a valid `MONGO_URI` in your backend `.env` file.

### 2. Start the Backend

```bash
# Option 1: Using the startup script (handles port conflicts)
python3 start_backend.py

# Option 2: Directly
cd backend
uvicorn main:app --reload --port 8001
```

### 3. Start the Frontend

```bash
cd app
npm install
npm run dev
```

The frontend will start on `http://localhost:5173`.

### Build for Production

```bash
cd app
npm run build
```

## NLP Pipeline & Logic

The system utilizes an enterprise-grade NLP matching pipeline:
1. **Extraction Pipeline**: Tries digital layer extraction (`pdfplumber`). If `< 50 chars` are found, dynamically falls back to rendering PDF images to array (`PyMuPDF`) and running Neural OCR (`EasyOCR`).
2. **Text Normalization**: Comprehensive alias resolution mapped across various software/business terminology.
3. **Semantic BERT Matching**: Matches job requirements to candidate skills via cosine similarity embeddings. Evaluates partial credit using domain-specific dependency mapping and explicit bidirectional semantic skill groups.
4. **Scoring Formula**: Final similarity scores calculate a blend of full-text semantic similarity, specific skill match ratios, Title Match bonuses, and Seniority Alignment heuristics.

## License

MIT License

## Author

[AnkitKumar729](https://github.com/AnkitKumar729/IntelliHire)
