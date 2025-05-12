"""
OCR Mapper Module

This module provides abstract base classes and implementations for mapping OCR results
to standardized formats. It supports multiple output formats and includes validation
and error handling mechanisms.

Classes:
    OCRMapper: Abstract base class for OCR result mapping
    MapperValidationError: Custom exception for mapper validation failures
    MappingError: Custom exception for mapping operation failures
"""

from abc import ABC, abstractmethod

from google_format import GoogleVisionMapper
# from openai_format import OpenAIMapper
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src/utils')))
from config import setup_logger

from typing import Dict, List, Any
import logging
from .google_format import GoogleVisionMapper
from .utils.format_registry import FormatRegistry
from .utils.validation_pipeline import ValidationPipeline
from .utils.result_cache import ResultCache
from .utils.error_handler import ErrorHandler

logger = logging.getLogger(__name__)

class MapperValidationError(Exception):
    """Custom exception for mapper validation failures."""
    pass

class MappingError(Exception):
    """Custom exception for mapping operation failures."""
    pass


logger = setup_logger('pipeline')
class OCRMapper(ABC):
    """
    Abstract base class for OCR result mapping.

    This class defines the interface for mapping OCR results to standardized formats.
    It includes validation and error handling mechanisms.

    Attributes:
        google_vision_mapper (GoogleVisionMapper): Mapper for Google Vision API format
        format_registry (FormatRegistry): Registry for supported output formats
        validation_pipeline (ValidationPipeline): Pipeline for validating OCR results
        result_cache (ResultCache): Cache for storing mapped results
        error_handler (ErrorHandler): Handler for managing mapping errors

    Methods:
        extract_ocr_properties: Abstract method for extracting OCR properties
        map_ocr_result: Map OCR results to specified format
        _validate_mappers: Validate mapper configurations
    """

    def __init__(self):
        """Initialize the OCRMapper with required mappers."""
        self.google_vision_mapper = GoogleVisionMapper()
        self._validate_mappers()
        
        # Add format registry
        self.format_registry = FormatRegistry()
        
        # Add validation pipeline
        self.validation_pipeline = ValidationPipeline()
        
        # Add result caching
        self.result_cache = ResultCache()
        
        # Add error recovery
        self.error_handler = ErrorHandler()

    @abstractmethod
    def extract_ocr_properties(self, ocr_result: List[Any]) -> Dict[str, Any]:
        """
        Extract text, coordinates, and confidence from OCR result.

        Args:
            ocr_result (List[Any]): Raw OCR result from specific OCR model

        Returns:
            Dict[str, Any]: Dictionary containing extracted properties

        Raises:
            NotImplementedError: If method is not implemented by subclass
        """
        pass


    def _validate_mappers(self) -> None:
        """
        Validate mapper configurations.

        Raises:
            MapperValidationError: If required mapper methods are missing
        """
        if not hasattr(self.google_vision_mapper, 'map_to_google_vision'):
            raise MapperValidationError("GoogleVisionMapper missing required method")
    
    @logger.catch
    def map_ocr_result(self, ocr_result: Dict[str, Any], format_type: str = "google") -> Dict[str, Any]:
        """
        Map OCR result to specified format with error handling.

        Args:
            ocr_result (Dict[str, Any]): OCR result to map
            format_type (str): Target format type (default: "google")

        Returns:
            Dict[str, Any]: Mapped OCR result

        Raises:
            ValueError: If format type is not supported
            MappingError: If mapping operation fails
        """
        try:
            logger.info("Mapping ocr result")
            extracted_ocr = self.extract_ocr_properties(ocr_result)
            if format_type == "google":
                return self.google_vision_mapper.map_to_google_vision(extracted_ocr)
            else:
                raise ValueError(f"Unsupported format type: {format_type}")
        except Exception as e:
            logger.error(f"Mapping failed: {e}")
            raise MappingError(f"Failed to map OCR result: {str(e)}")


