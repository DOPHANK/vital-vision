"""
LLM Integration Module

This module provides integration with Large Language Models (LLMs) using the Llama model.
It includes functionality for model loading, text generation, and LangChain integration.

Classes:
    CustomLlamaLLM: Custom LLM implementation for Llama models
    LlamaIntegrationPipeline: Pipeline for Llama model integration
    ModelNotFoundError: Custom exception for model loading failures
    ModelLoadingError: Custom exception for model initialization failures
"""

import torch
from transformers import (
    LlamaForCausalLM, 
    LlamaTokenizer,
    PreTrainedTokenizerFast,
    AutoTokenizer
)
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os
import transformers

# Load environment variables from .env file
load_dotenv()

class ModelNotFoundError(Exception):
    """Custom exception for model loading failures."""
    pass

class ModelLoadingError(Exception):
    """Custom exception for model initialization failures."""
    pass

class CustomLlamaLLM:
    """
    Custom LLM implementation for Llama models.

    This class provides a custom implementation of the LLM interface for Llama models,
    handling model initialization and text generation.

    Attributes:
        model (Any): The Llama model instance
        tokenizer (Any): The tokenizer for the model
        device (str): Device to run the model on

    Methods:
        generate: Generate text from input prompt
    """

    def __init__(self, model: Any, tokenizer: Any, device: str = 'cpu'):
        """
        Initialize the custom LLM.

        Args:
            model (Any): Llama model instance
            tokenizer (Any): Model tokenizer
            device (str): Device to run on (default: 'cpu')
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def generate(self, prompt: str) -> str:
        """
        Generate text from input prompt.

        Args:
            prompt (str): Input text prompt

        Returns:
            str: Generated text
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(**inputs)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


def load_llama_model_from_checkpoint(checkpoint_path, model_name, device='cpu'):
    """
    Load the LLAMA model from a .pth checkpoint file with authentication using environment variables.

    Parameters:
    -----------
    checkpoint_path : str
        The path to the .pth file containing the model checkpoint.
    model_name : str
        The correct Hugging Face model identifier.
    device : str
        The device to run the model on ('cpu' or 'cuda').

    Returns:
    --------
    model : LlamaForCausalLM
        The LLAMA model loaded from the checkpoint.
    tokenizer : LlamaTokenizer
        The tokenizer for the LLAMA model.
    """
    # Get the token from environment variables
    token = os.getenv("HUGGINGFACE_TOKEN")
    
    # Load the model architecture
    # Load the tokenizer and model architecture from the Hugging Face model hub
    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_name, token=token)
    model = LlamaForCausalLM.from_pretrained(model_name, token=token,
                                             torch_dtype=torch.float16,
                                             low_cpu_mem_usage=True)
    
    # Load the weights from the checkpoint
    print("Init complete")
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict,strict=False)
    print("Model Loaded!")
    # model_directory = '/media/data/Workspace/vital-ocr/model_checkpoints/Meta-Llama-3.1-8B/original/'

    # Load the tokenizer and model
    # tokenizer = LlamaTokenizer.from_pretrained(model_directory)
    # model = LlamaForCausalLM.from_pretrained(model_directory,
    #                                          torch_dtype=torch.float16,
    #                                          low_cpu_mem_usage=True)

    # Example usage
    # input_text = "This is an example input text from OCR."
    # inputs = tokenizer(input_text, return_tensors="pt")

    # Generate output
    # outputs = model.generate(**inputs)
    # print(tokenizer.decode(outputs[0], skip_special_tokens=True))
    
    # print("Model Loaded!")
    
    # Load the weights from the checkpoint
    # model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    
    # Load the tokenizer
    # tokenizer = LlamaTokenizer.from_pretrained(model_name, use_auth_token=token)
    
    # Move the model to the specified device
    model.to(device)
    
    return model, tokenizer


def integrate_with_langchain(text_input, model, tokenizer, device='cpu',max_length=500, max_new_tokens=250):
    """
    Integrate the custom LLAMA model with LangChain to process text input.

    Parameters:
    -----------
    text_input : str
        The text input to process.
    model : LlamaForCausalLM
        The LLAMA model loaded from the checkpoint.
    tokenizer : LlamaTokenizer
        The tokenizer for the LLAMA model.
    device : str
        The device to run the model on ('cpu' or 'cuda').

    Returns:
    --------
    output : str
        The processed output from the LLAMA model.
    """
    custom_llama = CustomLlamaLLM(model=model, tokenizer=tokenizer, device=device)

    # Directly use the custom_llama to generate text
    output = custom_llama.generate(text_input)
    
    return output



# Load environment variables from .env file
load_dotenv()

# class LlamaIntegrationPipeline:
#     def __init__(self, model_id="meta-llama/Meta-Llama-3.1-8B", device="auto",max_length=500, max_new_tokens=250):
#         """
#         Initialize the LlamaIntegrationPipeline with the given model ID and device setup.

#         Parameters:
#         -----------
#         model_id : str
#             The Hugging Face model identifier for the LLAMA model.
#         device : str
#             The device setup, either "auto", "cpu", or "cuda".
#         """
#         # Get the Hugging Face token from environment variables
#         token = os.getenv("HUGGINGFACE_TOKEN")
        
#         # Set up the text generation pipeline using the specified model
#         self.pipeline = transformers.pipeline(
#             "text-generation",
#             model=model_id,
#             model_kwargs={"torch_dtype": torch.bfloat16, "use_auth_token": token},
#             device_map=device
#         )
#         # Store the max_length and max_new_tokens
#         self.max_length = max_length
#         self.max_new_tokens = max_new_tokens

#     def generate_text(self, prompt: str) -> str:
#         """
#         Generate text based on the provided prompt using the LLAMA model.

#         Parameters:
#         -----------
#         prompt : str
#             The input text prompt to generate text from.

#         Returns:
#         --------
#         str
#             The generated text.
#         """
#         result = self.pipeline(
#             prompt,
#             max_length=self.max_length,
#             max_new_tokens=self.max_new_tokens,
#             return_full_text=False
#         )
#         return result[0]['generated_text']


# def integrate_with_llama(text_input):
#     """
#     Integrate the custom LLAMA pipeline to process text input.

#     Parameters:
#     -----------
#     text_input : str
#         The text input to process.

#     Returns:
#     --------
#     output : str
#         The processed output from the LLAMA model.
#     """
#     # Initialize the LLAMA integration pipeline
#     llama_pipeline = LlamaIntegrationPipeline(max_length=308, max_new_tokens=50)

#     # Generate text using the LLAMA model
#     output = llama_pipeline.generate_text(text_input)
    
#     return output


class LlamaIntegrationPipeline:
    def __init__(self, model_dir: str, device: str = "cpu", 
                 max_length: int = 50, max_new_tokens: int = 30):
        # Add model versioning
        self.model_version = ModelVersioning()
        
        # Add quantization support
        self.quantization = QuantizationManager()
        
        # Add model optimization
        self.optimizer = ModelOptimizer()
        
        # Add memory management
        self.memory_manager = MemoryManager()
        
        self.model_dir = model_dir
        self.device = device
        self.max_length = max_length
        self.max_new_tokens = max_new_tokens
        
        # Add model loading validation
        self._validate_model_dir()
        self._load_model()
        
    def _validate_model_dir(self):
        """Validate model directory and files."""
        if not os.path.exists(self.model_dir):
            raise ModelNotFoundError(f"Model directory not found: {self.model_dir}")
            
    def _load_model(self):
        """Load model with proper error handling."""
        try:
            self.tokenizer = LlamaTokenizer.from_pretrained(self.model_dir)
            self.model = LlamaForCausalLM.from_pretrained(
                self.model_dir, 
                torch_dtype=torch.bfloat16, 
                low_cpu_mem_usage=True
            )
            self.model.to(self.device)
        except Exception as e:
            raise ModelLoadingError(f"Failed to load model: {str(e)}")

    def generate_text(self, prompt: str) -> str:
        """
        Generate text based on the provided prompt using the LLAMA model.

        Parameters:
        -----------
        prompt : str
            The input text prompt to generate text from.

        Returns:
        --------
        str
            The generated text.
        """
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Generate output
        outputs = self.model.generate(
            **inputs,
            max_length=self.max_length,
            max_new_tokens=self.max_new_tokens
        )
        
        # Decode the output
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text


def integrate_with_llama(text_input, model_dir):
    """
    Integrate the custom LLAMA pipeline to process text input.

    Parameters:
    -----------
    text_input : str
        The text input to process.
    model_dir : str
        The directory where the local model checkpoint is stored.

    Returns:
    --------
    output : str
        The processed output from the LLAMA model.
    """
    # Initialize the LLAMA integration pipeline with the local model directory
    llama_pipeline = LlamaIntegrationPipeline(model_dir=model_dir, max_length=308, max_new_tokens=50)

    # Generate text using the LLAMA model
    output = llama_pipeline.generate_text(text_input)
    
    return output

# Example usage in the script
if __name__ == "__main__":
    
    # model_dir = "/media/data/Workspace/vital-ocr/model_checkpoints/Meta-Llama-3.1-8B/original/"

    # input_text = "Please revise the content of this medical note written in Vietnamese."

    # # Integrate with LLAMA using the local checkpoint and generate the output
    # result = integrate_with_llama(input_text, model_dir)
    
    # # input_text = "Please revise the content of this medical note written in Vietnamese."
    # # input_text = "Please revise the content of this medical note writen in Vietnamese. Can CAl Thời gian Nhận định tình trạng duc_ gnh_da Wul?LQ 7e ducp Mê bel Bakh tiau 4a SQ0M0 B2C Aqaua Alaiu ZA #nÉ_qua_QL [Uloitu_zunclc BD r Ild_uba{ Juâc 4242 Ácuay 'Lat (ô4 Da qLdLADdL da JLal_claa 4ì2 à AliL [dd_chan 1@_ha Me Ju (2 ALqLA Jlaud Zc_qua ala Uel RLul IN L IBsoh_h28a ) xu4 Nan_du_ea2 22 [dhuiss Hax_Dxy 410  Llenhai Hls 22L 9P291 LQ Hôo_QeEmg qa Qiêt_ NB Annh vQ Vê Chlohcxidine 40 # That sành 43# 2Cing kucoỳ C2 30 464 442 LLLL ax1d4 Jleubyf 2ucn / Jluàadt gu421 day dÊd zqử AULl bPsDandccl Auug dgR 7v +l'ng {lulel_ Aq+4 1B Mzy Jop 112+"

    # # result = integrate_with_llama(input_text)
    
    # # Print the result
    # print(result)
    
    model_dir = "/media/data/Workspace/vital-ocr/model_checkpoints/Meta-Llama-3.1-8B/original/"

    # Load the tokenizer
    tokenizer = LlamaTokenizer.from_pretrained(model_dir)

    # Load the model
    model = LlamaForCausalLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16)



# Integrate with Langchain
# if __name__ == "__main__":
#     # file_path = '/media/data/Workspace/vital-ocr/model_checkpoints/Meta-Llama-3.1-8B/original/'
#     # print(os.path.exists(file_path))  
    
#     checkpoint_path = "/media/data/Workspace/vital-ocr/model_checkpoints/Meta-Llama-3.1-8B/original/consolidated.00.pth"  # Replace with the actual path to your .pth file
#     model_name = "meta-llama/Meta-Llama-3.1-8B"  # Replace with the correct model identifier
#     model, tokenizer = load_llama_model_from_checkpoint(checkpoint_path,
#                                                         model_name=model_name,
#                                                         device='cpu')
    
#     input_text = "Please revise the content of this medical note writen in Vietnamese. \
#         Can CAl Thời gian Nhận định tình trạng duc_ gnh_da Wul?LQ 7e ducp Mê bel Bakh tiau 4a SQ0M0 B2C Aqaua Alaiu ZA #nÉ_qua_QL [Uloitu_zunclc BD r Ild_uba{ Juâc 4242 Ácuay 'Lat (ô4 Da qLdLADdL da JLal_claa 4ì2 à AliL [dd_chan 1@_ha Me Ju (2 ALqLA Jlaud Zc_qua ala Uel RLul IN L IBsoh_h28a ) xu4 Nan_du_ea2 22 [dhuiss Hax_Dxy 410  Llenhai Hls 22L 9P291 LQ Hôo_QeEmg qa Qiêt_ NB Annh vQ Vê Chlohcxidine 40 # That sành 43# 2Cing kucoỳ C2 30 464 442 LLLL ax1d4 Jleubyf 2ucn / Jluàadt gu421 day dÊd zqử AULl bPsDandccl Auug dgR 7v +l'ng {lulel_ Aq+4 1B Mzy Jop 112+"
#     result = integrate_with_langchain(input_text)
#     print(result)
    