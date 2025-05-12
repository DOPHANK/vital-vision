# src/ocr/ocr_service.py

import easyocr
import paddleocr
import keras_ocr
from doctr.models import ocr_predictor
import pytesseract
import keras_ocr
from typing import List, Union, Dict, Any
from loguru import logger
from ..pipelines.langchain.llama_integration import integrate_with_langchain
import time
from ..utils.model_registry import ModelRegistry
from ..utils.performance_metrics import PerformanceMetrics
from ..utils.model_cache import ModelCache

# Define supported models and languages
SUPPORTED_MODELS = ['easyocr', 'doctr', 'kerasocr', 'pytesseract']
SUPPORTED_LANGUAGES = ['en', 'th', 'vi']

class OCRInitializationError(Exception):
    """Custom exception for OCR initialization failures."""
    pass

class OCRProcessingError(Exception):
    """Custom exception for OCR processing failures."""
    pass

class OCRService:
    """
    A unified interface for OCR operations using various OCR engines.

    This class provides a standardized way to perform OCR operations using different
    OCR engines while handling common operations like caching, retries, and error handling.

    Attributes:
        model_name (str): The name of the OCR model being used
        language (List[str]): List of language codes supported by the OCR model
        reader (object): The OCR reader instance
        cache (dict): Cache for storing OCR results
        max_retries (int): Maximum number of retry attempts for failed operations
        kwargs (dict): Additional configuration parameters
        model_registry (ModelRegistry): Model registry for managing OCR models
        metrics (PerformanceMetrics): Performance metrics for OCR operations
        async_enabled (bool): Whether async support is enabled
        batch_size (int): Batch size for processing
        model_cache (ModelCache): Model cache for storing OCR results

    Methods:
        __init__: Initialize the OCR service with specified model and languages
        extract_text: Extract text from images with caching and retry mechanism
        tune_parameters: Adjust model-specific parameters
        process_with_llama: Process OCR results using LLM
    """

    def __init__(self, model: str = 'easyocr', language: List[str] = ['en','th','vi'], **kwargs):
        """
        Initialize the OCR service.

        Args:
            model (str): OCR model to use (default: 'easyocr')
            language (List[str]): List of language codes (default: ['en','th','vi'])
            **kwargs: Additional configuration parameters

        Raises:
            ValueError: If model or language is not supported
            OCRInitializationError: If OCR service initialization fails
        """
        self.model_name = model.lower()
        self.language = language
        self.kwargs = kwargs
        self.cache = {}
        self.max_retries = kwargs.get('max_retries', 3)

        # Add model registry
        self.model_registry = ModelRegistry()
        
        # Add performance monitoring
        self.metrics = PerformanceMetrics()
        
        # Add async support
        self.async_enabled = kwargs.get('async_enabled', False)
        
        # Add batch processing capability
        self.batch_size = kwargs.get('batch_size', 1)
        
        # Add model caching
        self.model_cache = ModelCache()

        self._validate_config(model, language)
        try:
            self._initialize_ocr(model, language, **kwargs)
        except Exception as e:
            logger.error(f"Failed to initialize OCR service: {e}")
            raise OCRInitializationError(f"OCR initialization failed: {str(e)}")

    def _validate_config(self, model: str, language: List[str]) -> None:
        """
        Validate OCR configuration parameters.

        Args:
            model (str): OCR model name
            language (List[str]): List of language codes

        Raises:
            ValueError: If model or language is not supported
        """
        if model not in SUPPORTED_MODELS:
            raise ValueError(f"Unsupported OCR model: {model}")
        if not all(lang in SUPPORTED_LANGUAGES for lang in language):
            raise ValueError(f"Unsupported language in: {language}")

    def _initialize_ocr(self, model: str, language: List[str], **kwargs):
        try:
            if model == 'easyocr':
                self.reader = easyocr.Reader(language, **kwargs)
            # elif self.model_name == 'paddleocr':
            #     self.reader = paddleocr.OCR(lang=self.language[0])
            elif model == 'doctr':
                self.reader = ocr_predictor(pretrained=True)
            elif model == 'kerasocr':
                self.reader = keras_ocr.pipeline.Pipeline()
            elif model == 'pytesseract':
                extracted_text = pytesseract()
                logger.info(f"Pytesseract OCR Service initialized with languages: {self.language}")
                return extracted_text
            else:
                raise ValueError(f"OCR model '{model}' is not supported.")
            
            logger.info(f"OCR Service initialized with model: {model} and languages: {language}")

        except Exception as e:
            logger.error(f"Error initializing OCR Service: {e}")
            raise

    def extract_text(self, image_path: str, use_cache: bool = True) -> Union[str, List[str]]:
        """
        Extract text from an image with caching and retry mechanism.

        Args:
            image_path (str): Path to the image file
            use_cache (bool): Whether to use cached results (default: True)

        Returns:
            Union[str, List[str]]: Extracted text or list of text blocks

        Raises:
            OCRProcessingError: If text extraction fails after all retries
        """
        if use_cache and image_path in self.cache:
            return self.cache[image_path]
            
        for attempt in range(self.max_retries):
            try:
                result = self._extract_text_internal(image_path)
                if use_cache:
                    self.cache[image_path] = result
                return result
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise OCRProcessingError(f"Failed to process image after {self.max_retries} attempts: {e}")
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                time.sleep(1)  # Add delay between retries

    def _extract_text_internal(self, image_path: str) -> Union[str, List[str]]:
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
            # elif self.model_name == 'paddleocr':
            #     result = self.reader.ocr(image_path)
            #     extracted_text = " ".join([line[1][0] for line in result[0]])
            elif self.model_name == 'doctr':
                extracted_text = self.reader(image_path)
            elif self.model_name == 'kerasocr':
                extracted_text = self.reader.recognize(image_path)
            elif self.model_name == 'pytesseract':
                extracted_text = self.reader.image_to_string(image_path)
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
