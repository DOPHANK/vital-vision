# src/utils/logger.py

from loguru import logger

logger.add("ocr_service.log", rotation="10 MB")
