# NBFC Financial Analysis Chatbot

An intelligent chatbot system for analyzing NBFC (Non-Banking Financial Company) financial data, providing real-time query resolution, data aggregation, and graph generation capabilities.

## Features

- Real-time query resolution for financial data
- Automated data ingestion from PDF reports
- Intelligent text chunking and embedding
- Vector-based semantic search
- Structured data extraction and storage
- Dynamic graph generation
- MIS report generation

## System Requirements

- Python 3.10+
- 16GB RAM minimum
- 8-core CPU
- 500GB SSD storage
- Optional: GPU for faster LLM inference

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd nbfc-chatbot
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On Unix/MacOS
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download spaCy model:
```bash
python -m spacy download en_core_web_sm
```

5. Download Llama model:
```bash
# Follow instructions in scripts/download_model.py
```

## Project Structure

```
nbfc-chatbot/
├── app/
│   ├── main.py              # FastAPI application
│   ├── services/            # Core services
│   ├── routes/             # API routes
│   └── utils/              # Utility functions
├── data/
│   ├── pdfs/               # Raw PDF files
│   ├── raw/                # Extracted text
│   ├── chunks/             # Text chunks
│   ├── embeddings/         # FAISS index
│   └── finance.db          # SQLite database
├── scripts/
│   ├── extract_pdf_text.py # PDF text extraction
│   ├── chunk_text.py       # Text chunking
│   ├── create_embeddings.py # Embedding generation
│   └── extract_metrics.py  # Metrics extraction
├── tests/                  # Test files
└── docs/                   # Documentation
```

## Usage

1. Start the API server:
```bash
uvicorn app.main:app --reload
```

2. Access the API documentation:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

3. Run the data ingestion pipeline:
```bash
python scripts/extract_pdf_text.py
python scripts/chunk_text.py
python scripts/create_embeddings.py
python scripts/extract_metrics.py
```

## Development

- Follow PEP 8 style guide
- Write tests for new features
- Update documentation as needed
- Use logging for debugging

## License

[Your License Here]

## Contact

[Your Contact Information]
