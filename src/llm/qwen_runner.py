import os
import torch
import time
import logging
import colorlog
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from huggingface_hub import snapshot_download
from urllib.request import urlopen

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
CHECKPOINT_DIR = f"D:/Workspace/vital-vision/checkpoints/qwen_vl/{MODEL_NAME.split('/')[-1]}"
IMAGE_PATH = r"D:\Workspace\OpenLLM\photos\monitor_screen\01-11-2023 10.35.18.jpg"  # Raw string for Windows path
USE_FAST_PROCESSOR = True  # Enable fast processor
USE_FLASH_ATTENTION = False  # Disabled for CPU

# Function to download model if not available
def ensure_model_download(model_name, checkpoint_dir):
    model_path = os.path.join(checkpoint_dir, model_name.split("/")[-1])
    if not os.path.exists(model_path):
        logger.info(f"Model not found in {model_path}. Downloading to {checkpoint_dir}...")
        snapshot_download(repo_id=model_name, local_dir=model_path, local_dir_use_symlinks=False)
    else:
        logger.info(f"Using existing model in {model_path}. To force re-download, delete the folder.")
    return model_path

# Function to handle image input (local or URL)
def prepare_image(image_path):
    if image_path.startswith("http"):
        temp_dir = "D:/Workspace/vital-vision/temp_images"
        os.makedirs(temp_dir, exist_ok=True)
        local_path = os.path.join(temp_dir, os.path.basename(image_path))
        with urlopen(image_path) as response, open(local_path, "wb") as out_file:
            out_file.write(response.read())
        logger.info(f"Downloaded image to {local_path}")
        return local_path
    elif not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    return image_path

# Main execution
try:
    # Check device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Ensure model is downloaded
    model_path = ensure_model_download(MODEL_NAME, CHECKPOINT_DIR)

    # Load model
    try:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype="auto",
        )
        model = model.to(device)
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        logger.warning("Ensure Qwen2_5_VLForConditionalGeneration is defined in your transformers version. Try Qwen2VLForConditionalGeneration if this fails.")
        raise

    # Load processor
    try:
        processor = AutoProcessor.from_pretrained(
            model_path,
            use_fast=USE_FAST_PROCESSOR,
            min_pixels=256*28*28,  # Optimize for CPU
            max_pixels=1280*28*28,
        )
        logger.info(f"Processor loaded successfully (use_fast={USE_FAST_PROCESSOR}).")
    except Exception as e:
        logger.error(f"Failed to load processor: {e}")
        raise

    # Prepare image
    image_path = prepare_image(IMAGE_PATH)

    # Define messages with enhanced prompt for vital signs
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {
                    "type": "text",
                    "text": "Extract the vital signs from the monitor screen, including Blood Pressure (BP, mmHg), Heart Rate (HR, bpm), Respiratory Rate (RR, breaths/min), and Oxygen Saturation (SpO2, %). Format the output as: BP: X/Y mmHg, HR: Z bpm, RR: W breaths/min, SpO2: V%. Focus on clear numerical values."
                },
            ],
        }
    ]

    # Prepare inputs with timing
    logger.info("Starting processor steps...")
    
    # Measure chat template application
    start_time = time.time()
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    chat_template_time = time.time() - start_time
    logger.info(f"Chat template applied in {chat_template_time:.3f} seconds.")

    # Measure vision info processing
    start_time = time.time()
    image_inputs, video_inputs = process_vision_info(messages)
    vision_info_time = time.time() - start_time
    logger.info(f"Vision info processed in {vision_info_time:.3f} seconds.")

    # Measure input preparation
    start_time = time.time()
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    input_prep_time = time.time() - start_time
    logger.info(f"Inputs prepared in {input_prep_time:.3f} seconds.")

    # Total processor time
    total_processor_time = chat_template_time + vision_info_time + input_prep_time
    logger.info(f"Total processor time: {total_processor_time:.3f} seconds.")

    inputs = inputs.to(device)

    # Generate output
    logger.info("Starting inference...")
    start_time = time.time()
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    inference_time = time.time() - start_time
    logger.info(f"Inference completed in {inference_time:.3f} seconds.")

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    logger.info(f"Generated output: {output_text}")

except Exception as e:
    logger.error(f"Error: {e}")
    raise