# manual_test_ocr_service.py
import sys
import os
from pdf2image import convert_from_path, convert_from_bytes
from pdf2image.exceptions import (
    PDFInfoNotInstalledError,
    PDFPageCountError,
    PDFSyntaxError
)

# Add the parent directory of 'src' to sys.path
sys.path.append(os.getcwd())
from src.ocr.ocr_service import OCRService

# Example usage with Thai language
# ocr_service_thai = OCRService(model='easyocr', language=['th'])
# text_thai = ocr_service_thai.extract_text('tests/sample_images/thai_sample_image.png')
# print("Extracted Text (Thai):", text_thai)

# Example usage with Vietnamese language
# ocr_service_vietnamese = OCRService(model='easyocr', language=['vi'])
# text_vietnamese = ocr_service_vietnamese.extract_text('tests/sample_images/vietnamese_sample_image.jpg')
# print("Extracted Text (Vietnamese):", text_vietnamese)

# Example: Using EasyOCR with custom tuning
ocr_service_easyocr = OCRService(model='easyocr', language=['en'], gpu=True)
ocr_service_easyocr.tune_parameters(detector=True, recognizer=True)

CRF_PATH = "D:\\Workspace\\DiTi\\CRF_No.1-50\\CRF_No.1-50"
CRF_NAME = "0001-0001"

pages = convert_from_path(os.path.join(CRF_PATH, f"{CRF_NAME}.pdf"), 500)
for count, page in enumerate(pages):
    page.save(f'out{count}.jpg', 'JPEG')
    text_easyocr = ocr_service_easyocr.extract_text(f'out{count}.jpg')
    print("Extracted Text (EasyOCR):", text_easyocr)

# Example: Using PaddleOCR without specific tuning
# ocr_service_paddleocr = OCRService(model='paddleocr', language=['vi'])
# text_paddleocr = ocr_service_paddleocr.extract_text('tests/sample_images/vietnamese_sample_image.jpg')
# print("Extracted Text (PaddleOCR):", text_paddleocr)

# ocr_service_kerasocr = OCRService(model='pytesseract', language=['vi'])
# text_pytesseract = ocr_service_kerasocr.extract_text('tests/sample_images/health_record/IMG_9365.jpg')
# print("Extracted Text (pytesseract):", text_pytesseract)