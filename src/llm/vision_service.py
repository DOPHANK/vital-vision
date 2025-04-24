import os
import sys
import torch
import logging
import colorlog
from PIL import Image
from huggingface_hub import snapshot_download
from transformers import AutoProcessor, AutoModelForCausalLM
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from typing import Any, Tuple, Optional, Dict, List
import platform

try:
    import accelerate
    HAS_ACCELERATE = True
except ImportError:
    HAS_ACCELERATE = False
    logging.warning("accelerate not found. Installing required package...")
    import subprocess
    subprocess.check_call(["pip", "install", "accelerate"])
    import accelerate

sys.path.append("D:\Workspace\\vital-vision/Qwen-VL")
sys.path.append("D:\Workspace\\vital-vision/LLaVA")
from transformers import AutoProcessor as QwenVLProcessor
from transformers import AutoModelForVision2Seq as QwenVLForConditionalGeneration
from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM as LlavaForConditionalGeneration

from src.llm.cache_manager import CacheManager
from src.llm.config import ModelConfig, InferenceConfig

# Logger setup
log_formatter = colorlog.ColoredFormatter(
    "%(log_color)s%(asctime)s - %(levelname)s - %(message)s",
    log_colors={
        'DEBUG': 'cyan',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'bold_red',
    }
)
console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)
file_handler = logging.FileHandler("vision_extract.log")
file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

logging.basicConfig(level=logging.INFO, handlers=[console_handler, file_handler])
logger = logging.getLogger(__name__)

class ModelLoadingError(Exception):
    pass

class InferenceError(Exception):
    pass

def process_image_with_cache(
    image_path: str,
    prompt: str,
    cache_manager: CacheManager,
    model_config: ModelConfig,
    inference_config: InferenceConfig
) -> str:
    cache_key = cache_manager.get_cache_key(image_path, prompt)
    cached_result = cache_manager.get_cached_result(cache_key)
    if cached_result and inference_config.cache_results:
        return cached_result
        
    result = run_phi3(
        image_path,
        prompt,
        device="cpu",
        use_gpu=False
    )
    
    if inference_config.cache_results:
        cache_manager.cache_result(cache_key, result)
    
    return result

def get_device():
    if torch.cuda.is_available():
        vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        logger.info(f"Detected VRAM: {vram:.2f} GB")
        if vram < 8.1:
            logger.info("VRAM less than 8.1GB, falling back to CPU")
            return torch.device("cpu"), False
        return torch.device("cuda"), True
    logger.warning("No GPU found, using CPU.")
    return torch.device("cpu"), False

def load_image(image_path):
    try:
        return Image.open(image_path).convert("RGB")
    except Exception as e:
        logger.error(f"Error loading image: {e}")
        sys.exit(1)

def ensure_model(model_id):
    path = os.path.join(os.path.expanduser("~/.cache/huggingface/hub"), f"models--{model_id.replace('/', '--')}")
    if not os.path.exists(path):
        logger.warning(f"Downloading model: {model_id}")
        snapshot_download(repo_id=model_id)
    else:
        logger.info(f"Model {model_id} already cached.")

def load_model_and_processor(
    model_id: str,
    processor_cls: Any,
    model_cls: Any,
    device: str,
    use_gpu: bool
) -> Tuple[Any, Any]:
    try:
        processor = processor_cls.from_pretrained(
            model_id,
            trust_remote_code=True
        )
        
        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.float16 if use_gpu else torch.float32,
            "low_cpu_mem_usage": True,
            "use_cache": True,
            "attn_implementation": "eager",
            "use_flash_attention_2": False
        }
        
        logger.info(f"Loading model {model_id} on device {device}")
        model = model_cls.from_pretrained(
            model_id,
            **model_kwargs
        )
        
        model = model.to(device)
        logger.info(f"Model loaded successfully on {device}")
        
        return processor, model
        
    except Exception as e:
        logger.error(f"Failed to load model {model_id}: {str(e)}")
        raise ModelLoadingError(f"Failed to load model: {str(e)}")

def run_llava(image_path, prompt, device, use_gpu):
    model_id = "llava-hf/llava-1.5-7b-hf"
    
    try:
        # First ensure the model is downloaded
        ensure_model(model_id)
        
        # Use the correct processor and model classes for LLaVA
        from transformers import AutoProcessor, LlavaForConditionalGeneration
        
        processor = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=True
        )
        
        model = LlavaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if use_gpu else torch.float32,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        ).to(device)
        
        image = load_image(image_path)
        inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
        
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=200)
        
        return processor.decode(output[0], skip_special_tokens=True)
        
    except Exception as e:
        logger.error(f"Error in run_llava: {str(e)}")
        raise ModelLoadingError(f"Failed to run LLaVA: {str(e)}")

def run_phi3(
    image_path: str,
    prompt: str,
    device: str = "cpu",
    use_gpu: bool = False,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9
) -> str:
    model_id = "microsoft/Phi-3-vision-128k-instruct"
    
    try:
        processor, model = load_model_and_processor(
            model_id, AutoProcessor, AutoModelForCausalLM, device, use_gpu
        )
        
        logger.info("Processing input image and prompt")
        inputs = processor(
            images=image_path,
            text=prompt,
            return_tensors="pt"
        )
        
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        logger.info("Generating response")
        with torch.no_grad():
            try:
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    pad_token_id=processor.tokenizer.pad_token_id,
                    eos_token_id=processor.tokenizer.eos_token_id
                )
            except RuntimeError as e:
                if "out of memory" in str(e):
                    torch.cuda.empty_cache()
                    logger.warning("GPU out of memory, trying with smaller batch")
                    raise InferenceError("GPU out of memory. Try reducing batch size or model size.")
                raise
        
        response = processor.batch_decode(outputs, skip_special_tokens=True)[0]
        return response.strip()
        
    except ModelLoadingError as e:
        logger.error(f"Model loading error: {e}")
        raise
    except Exception as e:
        logger.error(f"Error during inference: {str(e)}")
        raise InferenceError(f"Inference failed: {str(e)}")

def run_qwen2vl(image_path, prompt, device, use_gpu):
    model_id = "Qwen/Qwen2-VL-7B-Instruct"
    processor, model = load_model_and_processor(model_id, QwenVLProcessor, QwenVLForConditionalGeneration, device, use_gpu)
    image = load_image(image_path)
    messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
    inputs = processor(messages=messages, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=200)
    return processor.decode(output[0], skip_special_tokens=True)

MODEL_DISPATCH = {
    "1": ("LLaVA-7B", run_llava),
    "2": ("Phi-3 Vision", run_phi3),
    "3": ("Qwen2-VL", run_qwen2vl),
}

def main():
    try:
        # logger.remove()
        # logger.add(
        #     sys.stdout,
        #     format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
        # )
        
        image_path = "data/01-11-2023 10.28.56.jpg"  # Replace with actual image path
        prompt = "Describe this image in detail."
        
        device, use_gpu = get_device()
        
        logger.info(f"Using device: {device}")
        logger.info(f"GPU enabled: {use_gpu}")
        
        logger.info("Starting inference")
        # result = run_phi3(image_path, prompt, device, use_gpu)
        # result = run_qwen2vl(image_path, prompt, device, use_gpu)
        result = run_llava(image_path, prompt, device, use_gpu)
        logger.info("Inference completed")
        print(f"\nModel output:\n{result}")
        
    except (ModelLoadingError, InferenceError) as e:
        logger.error(f"Error: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise

if __name__ == "__main__":
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    main()