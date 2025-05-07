import os
import torch
import time
import logging
import colorlog
import pandas as pd
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from huggingface_hub import snapshot_download

# Set up colorlog
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

# Configuration
MODEL_LIST = [
    "Qwen/Qwen2.5-VL-3B-Instruct",
    "Qwen/Qwen2.5-VL-7B-Instruct",
    "Qwen/Qwen2.5-VL-14B-Instruct",
    "Qwen/Qwen2.5-VL-32B-Instruct",
]
MODEL_NAME = MODEL_LIST[0]  # Lightweight model for CPU
CHECKPOINT_DIR = "./checkpoints/qwen_vl/"
IMAGE_PATH_FILE = "/media/nhatpth/DNV-USB/47/47EI/image.txt"  # Path to the file containing image paths
BASE_PATH = "/media/nhatpth/DNV-USB/47"  # Base path to prepend to relative paths
USE_FAST_PROCESSOR = True
USE_FLASH_ATTENTION = False

# Function to download model if not available
def ensure_model_download(model_name, checkpoint_dir):
    model_path = os.path.join(checkpoint_dir, model_name.split("/")[-1])
    if not os.path.exists(model_path):
        logger.info(f"Model not found in {model_path}. Downloading to {checkpoint_dir}...")
        snapshot_download(repo_id=model_name, local_dir=model_path, local_dir_use_symlinks=False)
    else:
        logger.info(f"Using existing model in {model_path}. To force re-download, delete the folder.")
    return model_path

# Function to read image paths from the text file
def read_image_paths(file_path):
    image_paths = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                path = parts[0]
                if path.startswith('/'):
                    path = path[1:]
                image_paths.append(path)
    return image_paths

# Function to process a single image
def process_image(image_path, model, processor, device):
    full_path = os.path.join(BASE_PATH, image_path)
    
    if not os.path.exists(full_path):
        logger.error(f"Image file not found: {full_path}")
        return {
            'image_path': image_path,
            'model_output': 'File not found',
            'inference_time': 0
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

        # Prepare inputs
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(device)

        # Generate output with timing
        start_time = time.time()
        generated_ids = model.generate(**inputs, max_new_tokens=128)
        inference_time = time.time() - start_time
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        return {
            'image_path': image_path,
            'model_output': output_text,
            'inference_time': round(inference_time, 3)
        }
    
    except Exception as e:
        logger.error(f"Error processing {image_path}: {e}")
        return {
            'image_path': image_path,
            'model_output': f'Error: {str(e)}',
            'inference_time': 0
        }

def main():
    try:
        # Check device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")

        output_dir = "./output"
        if not os.path.exists(output_dir):
            logger.info(f"Creating output directory: {output_dir}")
            os.makedirs(output_dir)

        # Ensure model is downloaded
        model_path = ensure_model_download(MODEL_NAME, CHECKPOINT_DIR)

        # Load model
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype="auto",
        ).to(device)
        logger.info("Model loaded successfully.")

        # Load processor
        processor = AutoProcessor.from_pretrained(
            model_path,
            use_fast=USE_FAST_PROCESSOR,
            min_pixels=256*28*28,  # Optimize for CPU
            max_pixels=1280*28*28,
        )
        logger.info(f"Processor loaded successfully (use_fast={USE_FAST_PROCESSOR}).")

        # Read image paths
        image_paths = read_image_paths(IMAGE_PATH_FILE)
        logger.info(f"Found {len(image_paths)} images to process")

        # Process each image
        results = []
        total_start_time = time.time()
        for image_path in tqdm(image_paths, desc="Processing images"):
            result = process_image(image_path, model, processor, device)
            results.append(result)
            
        total_time = time.time() - total_start_time
        logger.info(f"Total processing time: {total_time:.2f} seconds for {len(image_paths)} images")
            
        # Save results to CSV
        csv_path = os.path.join(output_dir, "vital_signs_results.csv")
        pd.DataFrame(results).to_csv(csv_path, index=False)
        logger.info(f"Results saved to {csv_path}")

    except Exception as e:
        logger.error(f"Error: {e}")
        raise

if __name__ == "__main__":
    main()