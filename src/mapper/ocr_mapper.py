from abc import ABC, abstractmethod
from google_format import GoogleVisionMapper
# from openai_format import OpenAIMapper
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src/utils')))
from config import setup_logger

logger = setup_logger('pipeline')
class OCRMapper(ABC):
    def __init__(self):
        """
        Initialize the OCRMapper with a GoogleVisionMapper instance.
        """
        self.google_vision_mapper = GoogleVisionMapper()
        # self.openai_mapper = OpenAIMapper()
    @abstractmethod
    def extract_ocr_properties(self, ocr_result: list) -> dict:
        """
        Extract text, coordinates, and confidence from the OCR result.

        This abstract method should be implemented by subclasses for each specific OCR model.

        Args:
            ocr_result (list): The raw OCR result from a specific OCR model.

        Returns:
            dict: A dictionary containing extracted properties (text, coordinates, confidence).

        Raises:
            NotImplementedError: If the method is not implemented by a subclass.
        """
        pass
    @logger.catch
    def map_ocr_result(self, ocr_result: dict) -> dict:
        """
        Map the OCR result to Google Vision API format.

        This method extracts properties using the model-specific implementation
        and then delegates the final mapping to the GoogleVisionMapper.

        Args:
            ocr_result (dict): The raw OCR result from a specific OCR model.

        Returns:
            dict: The OCR result mapped to Google Vision API format.
        """
        logger.info("Mapping ocr result")
        # Extract properties from ocr model
        extracted_ocr = self.extract_ocr_properties(ocr_result)
        
        # Delegate the final mapping to the GoogleVisionMapper
        return self.google_vision_mapper.map_to_google_vision(extracted_ocr)

        # if format_type == "google":
        #     return self.google_vision_mapper.map_to_google_vision(ocr_result)
        # # Delegate the final mapping to the OpenAIMapper
        # elif format_type == "openai":
        #     return self.openai_mapper.map_to_openai_format(ocr_result)
        # else:
        #     raise ValueError("Unsupported format type.")


		
		

        

        
