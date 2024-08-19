from fastapi import FastAPI, UploadFile, File 
from src.ocr.ocr_service import OCRService
from ..pipelines.langchain.llama_integration import integrate_with_langchain, load_llama_model_from_checkpoint


app = FastAPI()

# Load the model and tokenizer from the checkpoint
checkpoint_path = "model_checkpoints/Meta-Llama-3.1-8B-Instruct/original/consolidated.00.pth"  # Replace with the actual path to your .pth file
model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"  # Replace with the correct model identifier
model, tokenizer = load_llama_model_from_checkpoint(checkpoint_path,
                                                    model_name=model_name,
                                                    device='cpu')

# Initialize the OCR service
ocr_service = OCRService(model='easyocr', language=['en', 'th'])

@app.get("/")
async def root():
    return {"message": "OCR-LLM System Running"}


@app.post("/extract_text/")
async def extract_text(file: UploadFile = File(...)):
    # Save the file temporarily
    image_path = f"/tmp/{file.filename}"
    with open(image_path, "wb") as image:
        content = await file.read()
        image.write(content)
    
    # Extract text using OCR service
    extracted_text = ocr_service.extract_text(image_path)
    return {"extracted_text": extracted_text}

@app.post("/process_image/")
async def process_image(file: UploadFile = File(...)):
    # Save the file temporarily
    image_path = f"/tmp/{file.filename}"
    with open(image_path, "wb") as image:
        content = await file.read()
        image.write(content)
    
    # Extract text using OCR service
    # result = ocr_service.process_with_llama(image_path)
    # return {"result": result}

     
    # Extract text using OCR
    extracted_text = ocr_service.extract_text(image_path)
    
    # Process the text with LLAMA
    result = integrate_with_langchain(extracted_text, model, tokenizer)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)