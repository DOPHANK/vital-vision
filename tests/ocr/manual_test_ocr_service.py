# manual_test_ocr_service.py

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
ocr_service_easyocr = OCRService(model='easyocr', language=['en', 'vi'], gpu=False)
ocr_service_easyocr.tune_parameters(detector=True, recognizer=True)
text_easyocr = ocr_service_easyocr.extract_text('tests/sample_images/vietnamese_sample_image.jpg')
print("Extracted Text (EasyOCR):", text_easyocr)

# Example: Using PaddleOCR without specific tuning
ocr_service_paddleocr = OCRService(model='paddleocr', language=['vi'])
text_paddleocr = ocr_service_paddleocr.extract_text('tests/sample_images/vietnamese_sample_image.jpg')
print("Extracted Text (PaddleOCR):", text_paddleocr)
