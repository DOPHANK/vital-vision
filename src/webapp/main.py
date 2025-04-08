from datetime import datetime, timezone
import json
import os
from typing import Generator
from fastapi import Depends, FastAPI, HTTPException, UploadFile, File
from fastapi.concurrency import asynccontextmanager 
from src.ocr.ocr_service import OCRService
from ..pipelines.langchain.llama_integration import integrate_with_langchain, load_llama_model_from_checkpoint
from ..mapper.post_process import load_model_and_tokenizer, OCRPostProcessor, read_model_config
from ..database.models import db_manager, Request, Process, Response, Model
from sqlalchemy.orm import Session
from loguru import logger

# Database Dependency
def get_db() -> Generator[Session, None, None]:
    yield from db_manager.get_session()

# Load the model and tokenizer from the checkpoint
# checkpoint_path = "model_checkpoints/Meta-Llama-3.1-8B-Instruct/original/consolidated.00.pth"  # Replace with the actual path to your .pth file
# model, tokenizer = load_llama_model_from_checkpoint(checkpoint_path,
#                                                     model_name=model_name,
#                                                     device='cpu')
model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
llm_model = None
tokenizer = None
processor = None
ocr_service = None
config = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize your resources here
    global llm_model, tokenizer, processor, ocr_service, config
    
    # Initialize Llama and configurations
    logger.info("Loading model and tokenizer...")
    llm_model, tokenizer = load_model_and_tokenizer(model_name)
    processor = OCRPostProcessor(llm_model, tokenizer) #llama with langchain
    config = read_model_config() #paramater: src\mapper\model_cache\models--meta-llama--Meta-Llama-3.1-8B-Instruct\snapshots\0e9e39f249a16976918f6564b8830bc894c89659\generation_config.json
    
    # Initialize OCR service
    logger.info("Initializing OCR service...")
    ocr_service = OCRService(model='easyocr', language=['en', 'th'])
    
    # Initialize database
    logger.info("Initializing database...")
    db_manager.init_db()
    
    # Initialize default models in the database
    with next(get_db()) as db:
        try:
            # Check if models already exist
            if not db.query(Model).filter_by(name='easyocr').first():
                ocr_model = Model( name='easyocr', type='ocr', parameters=json.dumps({'languages': ['en', 'th']}), is_active=True )
                db.add(ocr_model)
            
            if not db.query(Model).filter_by(name='meta-llama').first():
                llm_model = Model( name='meta-llama', type='llm', parameters= config, is_active=True )
                db.add(llm_model)
            
            db.commit()
            logger.info("Database initialization completed successfully")
        except Exception as e:
            logger.error(f"Error during database initialization: {str(e)}")
            db.rollback()
            raise

    yield

    # Shutdown: Clean up your resources here
    logger.info("Shutting down application...")

app = FastAPI(lifespan=lifespan)

@app.get("/")
async def root():
    return {"message": "OCR-LLM System Running"}


@app.post("/extract_text/")
async def extract_text(file: UploadFile = File(...), db: Session = Depends(get_db)):
    try:
        # Save the file temporarily
        image_path = f"/tmp/{file.filename}"
        with open(image_path, "wb") as image:
            content = await file.read()
            image.write(content)
    
        # Create request record
        request = Request(
            file_path=image_path,
            file_name=file.filename,
            file_size=len(content),
            file_type=file.content_type,
            status="processing"
        )
        db.add(request)
        db.flush()  # Get the request ID
        
        # Get OCR model from database
        ocr_model = db.query(Model).filter_by(name='easyocr', type='ocr').first()
        # Create OCR process record
        ocr_process = Process(request_id=request.id, model_id=ocr_model.id, process_type='ocr', status="processing")
        db.add(ocr_process)
        db.flush() # Get the process ID

        # Extract text using OCR service
        extracted_text = ocr_service.extract_text(image_path)

        # Create response record
        ocr_response = Response( process_id=ocr_process.id, content=str(extracted_text) )
        db.add(ocr_response)
        
        # Update status
        ocr_process.status = "completed"
        ocr_process.completed_at = datetime.now(timezone.utc)
        request.status = "completed"
        
        db.commit()

        return {"extracted_text": extracted_text}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(image_path):  
            os.remove(image_path)

@app.post("/process_image/")
async def process_image(file: UploadFile = File(...), db: Session = Depends(get_db)):
    try:
        # Save the file temporarily
        image_path = f"/tmp/{file.filename}"
        with open(image_path, "wb") as image:
            content = await file.read()
            image.write(content)
        
        # Create request record
        request = Request(file_path=image_path, file_name=file.filename, file_size=len(content), file_type=file.content_type, status="processing")
        db.add(request)
        db.flush()  # Get the request ID
        
        # Get model from database
        ocr_model = db.query(Model).filter_by(name='easyocr', type='ocr').first()
        llm_model = db.query(Model).filter_by(name='meta-llama', type='llm').first()
        # Create OCR process record
        ocr_process = Process(request_id=request.id, model_id=ocr_model.id, process_type='ocr', status="processing")
        db.add(ocr_process)
        db.flush() # Get the process ID
        
        # Extract text using OCR
        extracted_text = ocr_service.extract_text(image_path)
        
        # Create OCR response record
        ocr_response = Response( process_id=ocr_process.id, content=str(extracted_text))
        db.add(ocr_response)
        # Update OCR process status
        ocr_process.status = "completed"
        ocr_process.completed_at = datetime.now(timezone.utc)

        # Create LLM process record
        llm_process = Process( request_id=request.id, model_id=llm_model.id, process_type='llm', status="processing" )
        db.add(llm_process)
        db.flush() 
        # Process the text with LLAMA
        # result = integrate_with_langchain(extracted_text, model, tokenizer)
        result = processor.process_text(extracted_text)

        # Create LLM response record
        llm_response = Response( process_id=llm_process.id, content=str(result) )
        db.add(llm_response)
        
        # Update LLM process status
        llm_process.status = "completed"
        llm_process.completed_at = datetime.now(timezone.utc)
        
        # Update request status
        request.status = "completed"
    
        db.commit()
        return {"processed_text": result}
    
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(image_path):
            os.remove(image_path)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)