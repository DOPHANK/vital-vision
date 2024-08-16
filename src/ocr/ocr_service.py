# src/ocr/ocr_service.py

import easyocr
import paddleocr
from typing import List, Union
from loguru import logger
from ..pipelines.langchain.llama_integration import integrate_with_langchain

class OCRService:
    """
    OCRService provides an interface to extract text from images using different OCR models.

    Attributes:
    -----------
    model_name : str
        The name of the OCR model being used ('easyocr' or 'paddleocr').
    language : List[str]
        A list of language codes supported by the OCR model.
    reader : object
        The OCR reader instance initialized based on the selected model.

    Methods:
    --------
    __init__(model: str = 'easyocr', language: List[str] = ['en'], **kwargs):
        Initializes the OCRService with the specified model, language, and optional tuning parameters.

    extract_text(image_path: str) -> Union[str, List[str]]:
        Extracts text from the given image file path.

    tune_parameters(**kwargs):
        Allows tuning of model-specific parameters.
    """

    def __init__(self, model: str = 'easyocr', language: List[str] = ['en','th','vi'], **kwargs):
        """
        Initializes the OCRService.

        Parameters:
        -----------
        model : str
            The OCR model to use ('easyocr' or 'paddleocr'). Default is 'easyocr'.
        language : List[str]
            A list of language codes to initialize the OCR model with. Default is ['en'].
            Supported languages include 'en' (English), 'th' (Thai), and 'vi' (Vietnamese).
        **kwargs : dict
            Additional tuning parameters for the OCR model.
        
        Raises:
        -------
        ValueError
            If an unsupported OCR model is specified.
        Exception
            If the OCR reader fails to initialize.
        """
        self.model_name = model.lower()
        self.language = language
        self.kwargs = kwargs

        try:
            if self.model_name == 'easyocr':
                self.reader = easyocr.Reader(self.language, **kwargs)
            elif self.model_name == 'paddleocr':
                self.reader = paddleocr.OCR(lang=self.language[0])
            else:
                raise ValueError(f"OCR model '{self.model_name}' is not supported.")
            
            logger.info(f"OCR Service initialized with model: {self.model_name} and languages: {self.language}")

        except Exception as e:
            logger.error(f"Error initializing OCR Service: {e}")
            raise

    def extract_text(self, image_path: str) -> Union[str, List[str]]:
        """
        Extracts text from the given image file path.

        Parameters:
        -----------
        image_path : str
            The file path to the image from which text needs to be extracted.
        
        Returns:
        --------
        Union[str, List[str]]
            The extracted text as a single string or a list of text blocks.
        
        Raises:
        -------
        Exception
            If text extraction fails.
        """
        try:
            if self.model_name == 'easyocr':
                result = self.reader.readtext(image_path, detail=0)
                extracted_text = " ".join(result)
            elif self.model_name == 'paddleocr':
                result = self.reader.ocr(image_path)
                extracted_text = " ".join([line[1][0] for line in result[0]])

            logger.info(f"Text extracted from {image_path} using {self.model_name}")
            return extracted_text

        except Exception as e:
            logger.error(f"Error extracting text from {image_path}: {e}")
            raise

    def tune_parameters(self, **kwargs):
        """
        Allows tuning of model-specific parameters.

        Parameters:
        -----------
        **kwargs : dict
            Key-value pairs of parameters to be tuned.
        
        Example:
        --------
        tune_parameters(confidence_threshold=0.5)
        """
        self.kwargs.update(kwargs)
        if self.model_name == 'easyocr':
            self.reader = easyocr.Reader(self.language, **self.kwargs)
            logger.info(f"EasyOCR parameters updated: {self.kwargs}")
        elif self.model_name == 'paddleocr':
            # PaddleOCR tuning can be added here if required.
            logger.info("PaddleOCR tuning not implemented.")

    def process_with_llama(self, image_path: str) -> Union[str, List[str]]:
        """
        Processes an image with OCR and then uses LLAMA to further process the text.
        
        Parameters:
        -----------
        image_path : str
            The path to the image file.
        
        Returns:
        --------
        str
            The final output after OCR and LLAMA processing.
        """
        extracted_text = self.extract_text(image_path)
        final_output = integrate_with_langchain(extracted_text)
        return final_output