import os
import torch
import time
import logging
import colorlog
import pandas as pd
import psutil
import argparse
from pathlib import Path
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from huggingface_hub import snapshot_download
from urllib.request import urlopen
import shutil
import pkg_resources

# Verify required libraries
try:
    import bitsandbytes
    bitsandbytes_version = pkg_resources.get_distribution("bitsandbytes").version
except ImportError:
    raise ImportError("bitsandbytes library is required. Install it using: pip install bitsandbytes")

try:
    import psutil
except ImportError:
    raise ImportError("psutil library is required for memory tracking. Install it using: pip install psutil")

# Set up colorlog (early for warnings)
handler = colorlog.StreamHandler()
handler.setFormatter(colorlog.ColoredFormatter(
    "%(log_color)s%(levelname)s:%(name)s:%(message)s",
    log_colors={
        'DEBUG': 'cyan',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'red,bg_white',
    }
))
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(handler)

# Check for BitsAndBytesConfig availability
BITSANDBYTES_AVAILABLE = False
try:
    from transformers import BitsAndBytesConfig
    BITSANDBYTES_AVAILABLE = True
    logger.info(f"bitsandbytes version {bitsandbytes_version} loaded successfully")
except ImportError:
    logger.warning(f"BitsAndBytesConfig not found in bitsandbytes (version {bitsandbytes_version}). 4-bit/8-bit quantization is disabled. Update with: pip install --upgrade bitsandbytes")

# Configuration
MODEL_LIST = [
    "Qwen/Qwen2.5-VL-3B-Instruct",
    "Qwen/Qwen2.5-VL-7B-Instruct",
    "Qwen/Qwen2.5-VL-14B-Instruct",
    "Qwen/Qwen2.5-VL-32B-Instruct",
]
DEFAULT_MODEL_NAME = MODEL_LIST[0]
CHECKPOINT_DIR = "./checkpoints/qwen_vl/"
IMAGE_PATH_FILE = "./47EI/image_path.txt"
BASE_PATH = "/home/nhatpth/Vital-Vision/vital-vision"
TEMP_DIR = "./temp_images"
OUTPUT_DIR = "./output"
USE_FAST_PROCESSOR = True
USE_FLASH_ATTENTION = False

# Command-line argument parsing
parser = argparse.ArgumentParser(description="Run Qwen2.5-VL model for vital signs extraction")
parser.add_argument("--model", choices=MODEL_LIST, default=DEFAULT_MODEL_NAME, help="Model to use")
parser.add_argument("--batch", action="store_true", default=True, help="Enable batch processing of 20 images (default)")
parser.add_argument("--image", type=str, help="Path or URL to single image for processing (disables batch mode)")
parser.add_argument("--quantization", choices=["4bit", "8bit", "16bit", "none"], default="none",
                    help="Quantization level (4bit, 8bit, 16bit, or none)")
parser.add_argument("--device", choices=["cuda", "cpu"], default="cpu", help="Device to use (cuda or cpu)")
args = parser.parse_args()

# Function to get memory usage
def get_memory_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    ram_usage = mem_info.rss / 1024**2  # Convert to MB
    gpu_usage = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() and args.device == "cuda" else 0
    return ram_usage, gpu_usage

# Function to download model if not available
def ensure_model_download(model_name, checkpoint_dir):
    model_path = Path(checkpoint_dir) / model_name.split("/")[-1]
    if not model_path.exists():
        logger.info(f"Model not found in {model_path}. Downloading to {checkpoint_dir}...")
        snapshot_download(repo_id=model_name, local_dir=model_path, local_dir_use_symlinks=False)
    else:
        logger.info(f"Using existing model in {model_path}. To force re-download, delete the folder.")
    return str(model_path)

# Function to handle image input (local or URL)
def prepare_image(image_path):
    if image_path.startswith("http"):
        temp_dir = Path(TEMP_DIR)
        temp_dir.mkdir(exist_ok=True)
        local_path = temp_dir / os.path.basename(image_path)
        with urlopen(image_path) as response, open(local_path, "wb") as out_file:
            out_file.write(response.read())
        logger.info(f"Downloaded image to {local_path}")
        return str(local_path)
    full_path = Path(image_path)
    if not full_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")
    return str(full_path)

# Function to read image paths from the text file
def read_image_paths(file_path, limit=20):
    image_paths = []
    file_path = Path(file_path)
    if not file_path.exists():
        logger.error(f"Image path file not found: {file_path}")
        return image_paths
    with open(file_path, 'r') as f:
        for line in f:
            if len(image_paths) >= limit:
                break
            parts = line.strip().split()
            if parts:
                path = parts[0]
                if path.startswith('/') and path.split('/')[-1] == 'attachment_1.jpg':
                    path = path[1:]
                image_paths.append(path)
    return image_paths

# Function to process a single image
def process_image(image_path, model, processor, device, is_batch=False):
    if is_batch:
        full_path = str(Path(BASE_PATH) / image_path)
    else:
        full_path = prepare_image(image_path)
    
    if not Path(full_path).exists():
        logger.error(f"Image file not found: {full_path}")
        return {
            'image_path': image_path,
            'model_output': 'File not found',
            'inference_time': 0,
            'chat_template_time': 0,
            'vision_info_time': 0,
            'input_prep_time': 0,
            'total_processor_time': 0,
            'ram_usage_mb': 0,
            'gpu_usage_mb': 0
        }
    
    try:
        # Define messages with prompt for vital signs
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": full_path},
                    {
                        "type": "text",
                        "text": "Extract the vital signs from the monitor screen, including Blood Pressure (BP, mmHg), Heart Rate (HR, bpm), Respiratory Rate (RR, breaths/min), and Oxygen Saturation (SpO2, %). Format the output as: BP: X/Y mmHg, HR: Z bpm, RR: W breaths/min, SpO2: V%. Focus on clear numerical values."
                    },
                ],
            }
        ]

        # Prepare inputs with timing
        start_time = time.time()
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        chat_template_time = time.time() - start_time

        start_time = time.time()
        image_inputs, video_inputs = process_vision_info(messages)
        vision_info_time = time.time() - start_time

        start_time = time.time()
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        input_prep_time = time.time() - start_time

        total_processor_time = chat_template_time + vision_info_time + input_prep_time
        inputs = inputs.to(device)

        # Generate output
        start_time = time.time()
        generated_ids = model.generate(**inputs, max_new_tokens=128)
        inference_time = time.time() - start_time

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        # Get memory usage
        ram_usage, gpu_usage = get_memory_usage()

        return {
            'image_path': image_path,
            'model_output': output_text,
            'inference_time': round(inference_time, 3),
            'chat_template_time': round(chat_template_time, 3),
            'vision_info_time': round(vision_info_time, 3),
            'input_prep_time': round(input_prep_time, 3),
            'total_processor_time': round(total_processor_time, 3),
            'ram_usage_mb': round(ram_usage, 2),
            'gpu_usage_mb': round(gpu_usage, 2)
        }
    
    except Exception as e:
        logger.error(f"Error processing {image_path}: {e}")
        return {
            'image_path': image_path,
            'model_output': f'Error: {str(e)}',
            'inference_time': 0,
            'chat_template_time': 0,
            'vision_info_time': 0,
            'input_prep_time': 0,
            'total_processor_time': 0,
            'ram_usage_mb': 0,
            'gpu_usage_mb': 0
        }

# Function to clean temporary directory
def clean_temp_dir(temp_dir):
    if Path(temp_dir).exists():
        shutil.rmtree(temp_dir)
        logger.info(f"Cleaned temporary directory: {temp_dir}")

def main():
    try:
        # If --image is provided, disable batch mode
        if args.image:
            args.batch = False
            logger.info("Single image mode enabled due to --image argument")

        # Set up device and quantization
        device = args.device
        logger.info(f"Using device: {device}")

        # Warn about CPU quantization limitations
        if args.device == "cpu" and args.quantization in ["4bit", "8bit"]:
            logger.warning("4-bit and 8-bit quantization are not supported on CPU. Falling back to no quantization.")
            args.quantization = "none"

        quantization_config = None
        if args.quantization != "none":
            if not BITSANDBYTES_AVAILABLE:
                logger.error(f"Quantization ({args.quantization}) requires BitsAndBytesConfig, which is unavailable in bitsandbytes (version {bitsandbytes_version}). Install a compatible version: pip install --upgrade bitsandbytes")
                raise ImportError("BitsAndBytesConfig not found in bitsandbytes")
            if args.quantization == "4bit":
                quantization_config = BitsAndBytesConfig(
                                                        load_in_4bit=True,
                                                        # bnb_4bit_quant_type="nf4",
                                                        # bnb_4bit_use_double_quant=True,
                                                        # bnb_4bit_compute_dtype=torch.bfloat16,
                                                        )
                logger.info("Using 4-bit quantization")
            elif args.quantization == "8bit":
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)
                logger.info("Using 8-bit quantization")
            elif args.quantization == "16bit":
                quantization_config = BitsAndBytesConfig(compute_dtype=torch.float16)
                logger.info("Using 16-bit quantization")

        # Create output directory
        output_dir = Path(OUTPUT_DIR)
        output_dir.mkdir(exist_ok=True)

        # Ensure model is downloaded
        model_path = ensure_model_download(args.model, CHECKPOINT_DIR)

        # Load model
        try:
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype="auto",
                quantization_config=quantization_config,
            )
            model = model.to(device)
            logger.info(f"Model {args.model} loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            logger.warning("Ensure Qwen2_5_VLForConditionalGeneration is defined. Try Qwen2VLForConditionalGeneration if this fails.")
            raise

        # Load processor
        try:
            processor = AutoProcessor.from_pretrained(
                model_path,
                use_fast=USE_FAST_PROCESSOR,
                min_pixels=256*28*28,
                max_pixels=1280*28*28,
            )
            logger.info(f"Processor loaded successfully (use_fast={USE_FAST_PROCESSOR}).")
        except Exception as e:
            logger.error(f"Failed to load processor: {e}")
            raise

        if args.batch:
            # Batch processing
            image_paths = read_image_paths(IMAGE_PATH_FILE)
            if not image_paths:
                logger.error("No images found to process. Check IMAGE_PATH_FILE.")
                return
            logger.info(f"Found {len(image_paths)} images to process")

            results = []
            total_start_time = time.time()
            for image_path in tqdm(image_paths, desc="Processing images"):
                result = process_image(image_path, model, processor, device, is_batch=True)
                results.append(result)
            
            total_time = time.time() - total_start_time
            logger.info(f"Total processing time: {total_time:.2f} seconds for {len(image_paths)} images")

            # Calculate averages
            valid_results = [r for r in results if r['model_output'] != 'File not found' and not r['model_output'].startswith('Error')]
            if valid_results:
                avg_inference = sum(r['inference_time'] for r in valid_results) / len(valid_results)
                avg_processor = sum(r['total_processor_time'] for r in valid_results) / len(valid_results)
                avg_ram = sum(r['ram_usage_mb'] for r in valid_results) / len(valid_results)
                avg_gpu = sum(r['gpu_usage_mb'] for r in valid_results) / len(valid_results)
                logger.info(f"Average inference time: {avg_inference:.3f} seconds")
                logger.info(f"Average processor time: {avg_processor:.3f} seconds")
                logger.info(f"Average RAM usage: {avg_ram:.2f} MB")
                logger.info(f"Average GPU usage: {avg_gpu:.2f} MB")

            # Save results to CSV
            csv_path = output_dir / "vital_signs_results.csv"
            pd.DataFrame(results).to_csv(csv_path, index=False)
            logger.info(f"Results saved to {csv_path}")
        else:
            # Single image processing
            result = process_image(args.image, model, processor, device, is_batch=False)
            logger.info(f"Generated output: {result['model_output']}")
            logger.info(f"Inference time: {result['inference_time']:.3f} seconds")
            logger.info(f"Chat template time: {result['chat_template_time']:.3f} seconds")
            logger.info(f"Vision info time: {result['vision_info_time']:.3f} seconds")
            logger.info(f"Input prep time: {result['input_prep_time']:.3f} seconds")
            logger.info(f"Total processor time: {result['total_processor_time']:.3f} seconds")
            logger.info(f"RAM usage: {result['ram_usage_mb']:.2f} MB")
            logger.info(f"GPU usage: {result['gpu_usage_mb']:.2f} MB")

        # Clean temporary directory
        clean_temp_dir(TEMP_DIR)

    except Exception as e:
        logger.error(f"Error: {e}")
        raise

if __name__ == "__main__":
    main()