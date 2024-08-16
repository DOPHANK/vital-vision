import unittest
from src.ocr.ocr_service import OCRService

class TestOCRService(unittest.TestCase):
    
    def test_easyocr_extraction(self):
        ocr_service = OCRService(model='easyocr', language=['en'])
        text = ocr_service.extract_text('tests/sample_images/sample_image.png')
        self.assertIsInstance(text, str)
        self.assertGreater(len(text), 0)

    def test_paddleocr_extraction(self):
        ocr_service = OCRService(model='paddleocr', language=['en'])
        text = ocr_service.extract_text('tests/sample_images/sample_image.png')
        self.assertIsInstance(text, str)
        self.assertGreater(len(text), 0)

    def test_thai_language_easyocr(self):
        ocr_service = OCRService(model='easyocr', language=['th'])
        text = ocr_service.extract_text('tests/sample_images/thai_sample_image.png')
        self.assertIsInstance(text, str)
        self.assertGreater(len(text), 0)

    def test_vietnamese_language_easyocr(self):
        ocr_service = OCRService(model='easyocr', language=['vi'])
        text = ocr_service.extract_text('tests/sample_images/vietnamese_sample_image.png')
        self.assertIsInstance(text, str)
        self.assertGreater(len(text), 0)

    def test_invalid_model(self):
        with self.assertRaises(ValueError):
            OCRService(model='invalid_model', language=['en'])

if __name__ == '__main__':
    unittest.main()
