
# OCR-to-LLM System

## Overview
This project is an OCR (Optical Character Recognition) to LLM (Language Model) processing system. It extracts text from images or documents, processes the text using a pre-trained language model (LLAMA 3.1), and returns the processed text. The system is designed to be modular, scalable, and easily deployable on an on-site server.

## Project Structure
```plaintext
project_root/
│
├── src/
│   ├── ocr/
│   │   ├── __init__.py
│   │   └── ocr_service.py
│   │
│   ├── llm/
│   │   ├── __init__.py
│   │   └── llama_service.py
│   │
│   ├── pipeline/
│   │   ├── __init__.py
│   │   └── pipeline_service.py
│   │
│   ├── webapp/
│   │   ├── __init__.py
│   │   └── main.py
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── logger.py
│   │   └── config.py
│   │
│   └── __init__.py
│
├── data/
│   ├── input_images/
│   └── processed_text/
│
├── tests/
│   ├── __init__.py
│   ├── test_ocr_service.py
│   ├── test_llama_service.py
│   ├── test_pipeline_service.py
│   └── test_routes.py
│
├── docs/
│   ├── requirements.txt
│   ├── README.md
│   └── architecture_diagram.png
│
├── scripts/
│   ├── deploy.sh
│   └── start_dev.sh
│
├── .env
├── .gitignore
├── Dockerfile
└── docker-compose.yml
```

## Requirements

### Python Packages
- Python 3.8+
- FastAPI
- Uvicorn
- Gunicorn
- EasyOCR
- PaddleOCR
- Transformers
- Celery
- Redis
- SQLAlchemy
- Psycopg2-binary
- Docker
- Docker Compose

### System Requirements
- Docker installed and running
- PostgreSQL database
- Redis for task queuing
- On-site server with sufficient resources for LLM processing

## Setup Instructions

### Step 1: Clone the Repository
```bash
git clone <repository-url>
cd ocr-to-llm
```

### Step 2: Set Up the Python Environment
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r docs/requirements.txt
```

### Step 3: Set Up the Database
Ensure PostgreSQL is installed and running. Create a database and user:
```sql
CREATE DATABASE ocr_llm_db;
CREATE USER ocr_user WITH ENCRYPTED PASSWORD 'your_password';
GRANT ALL PRIVILEGES ON DATABASE ocr_llm_db TO ocr_user;
```

### Step 4: Configure the Environment Variables
Create a `.env` file in the project root with the following contents:
```plaintext
DATABASE_URL=postgresql://ocr_user:your_password@localhost/ocr_llm_db
REDIS_URL=redis://localhost:6379/0
```

### Step 5: Run the Application Locally
```bash
uvicorn src.webapp.main:app --reload
```

### Step 6: Run Celery Worker
Start the Celery worker to handle background tasks:
```bash
celery -A src.pipeline.pipeline_service worker --loglevel=info
```

### Step 7: Dockerize the Application
To build and run the application in Docker containers, use:
```bash
docker-compose up --build
```

## Usage
- Access the application at `http://localhost:8000`.
- Upload images to extract text, which will be processed by the LLM.

## Testing
Run unit tests with:
```bash
pytest
```

## Contributing
Contributions are welcome! Please submit a pull request or open an issue to discuss any changes.

## License
This project is licensed under the MIT License.
