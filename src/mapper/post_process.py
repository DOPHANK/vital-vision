import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain.llms.base import LLM
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from typing import Any, Optional, List, Tuple
from pydantic import Field, PrivateAttr
import sys
import os
from dotenv import load_dotenv

load_dotenv()
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src/utils')))
from config import setup_logger


# Set up the logger for the LLM pipeline
logger = setup_logger('pipeline')

# Load the Llama model and tokenizer
model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"


# logger.info(f"Starting to load the model {model_name} into memory")

def load_model_and_tokenizer(model_name: str, cache_dir: str = "./model_cache") -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load a model and tokenizer from Hugging Face or a local path.
    """
    model_path = ""
    try:
        # Set up kwargs for model loading
        model_kwargs = {
            "torch_dtype": torch.bfloat16,
            "device_map": "auto"
        }
    
        if cache_dir:
            model_kwargs["cache_dir"] = cache_dir

        # Load the model
        if not os.path.exists(model_path):
            print(f"Downloading model {model_name}...")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                **model_kwargs,
                token=os.getenv("HUGGINGFACE_TOKEN")
            )
            # model.save_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_name, token=os.getenv("HUGGINGFACE_TOKEN") )
            # tokenizer.save_pretrained(model_path)
        else:
            print(f"Loading model from {model_path}...")
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                **model_kwargs,
            )
            tokenizer = AutoTokenizer.from_pretrained(model_path)

        return model, tokenizer

    except Exception as e:
        raise ValueError(f"Failed to load model and tokenizer: {str(e)}")


# Define the custom LLM class
class LlamaVitalSignsExtractor(LLM):
    model_name: str = Field(default=model_name)
    _model: Any = PrivateAttr()
    _tokenizer: Any = PrivateAttr()

    def __init__(self, model, tokenizer, **data):
        super().__init__(**data)
        self._model = model
        self._tokenizer = tokenizer
        logger.info(f"Initialized LlamaVitalSignsExtractor with model {self.model_name}")

    @property
    def _llm_type(self) -> str:
        return "llama_vital_signs_extractor"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        logger.debug(f"Calling LlamaVitalSignsExtractor with prompt: {prompt[:50]}...")
        try:
            inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
            logger.info("You are using: ", self._model.device)
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=100,
                pad_token_id=self._tokenizer.eos_token_id
            )
            text = self._tokenizer.decode(outputs[0], skip_special_tokens=True)

            if stop is not None:
                for s in stop:
                    if s in text:
                        text = text[:text.index(s)]

            logger.debug(f"Generated text: {text[:50]}...")
            return text[len(prompt):]
        except Exception as e:
            logger.error(f"Error in LlamaVitalSignsExtractor._call: {e}")
            raise


# Define the OCRPostProcessor class
class OCRPostProcessor:
    def __init__(self, llm=None):
        model, tokenizer = load_model_and_tokenizer(model_name)
        self.llm = llm if llm else LlamaVitalSignsExtractor(model=model, tokenizer=tokenizer)
        logger.info("Initialized OCRPostProcessor")

    def process_text(self, ocr_text: str) -> str:
        """Process the extracted OCR text using LangChain."""
        logger.info("Processing OCR text with LangChain")
        # Check for empty or invalid input
        if not ocr_text or ocr_text.strip() in ['', '[]']:
            logger.warning("Empty or invalid OCR text received. Returning empty result.")
            return ""
        prompt = PromptTemplate(input_variables=["ocr_text"], template="Extract vital sign data from the text: {ocr_text}")
        chain = LLMChain(llm=self.llm, prompt=prompt)

        try:
            result = chain.run({"ocr_text": ocr_text})
            logger.info("Successfully processed OCR text")
            logger.debug(f"Processed result: {result[:50]}...")
            return result
        except Exception as e:
            logger.error(f"Error processing OCR text: {e}")
            raise

# Example usage
# processor = OCRPostProcessor()
# test_text = "Patient vitals: BP 120/80, HR 72, Temp 98.6F, RR 16"
# result = processor.process_text(test_text)
# print(f"Extracted vital signs: {result}")


