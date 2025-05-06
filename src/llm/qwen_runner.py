import os
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from huggingface_hub import snapshot_download

# Define checkpoint folder for model storage
CHECKPOINT_DIR = "./checkpoints/qwen_vl"
MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"

# Function to download model if not available
def ensure_model_download(model_name, checkpoint_dir):
    model_path = os.path.join(checkpoint_dir, model_name.split("/")[-1])
    if not os.path.exists(model_path):
        print(f"Model not found in {model_path}. Downloading to {checkpoint_dir}...")
        snapshot_download(repo_id=model_name, local_dir=model_path, local_dir_use_symlinks=False)
    return model_path

# Ensure model is downloaded
model_path = ensure_model_download(MODEL_NAME, CHECKPOINT_DIR)

# Check if CUDA is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load the model
try:
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto",
        # Optional: Enable flash_attention_2 for better performance
        # attn_implementation="flash_attention_2",
    )
    print("Model loaded successfully.")
except Exception as e:
    raise Exception(f"Failed to load model: {e}")

# Load the processor
try:
    processor = AutoProcessor.from_pretrained(
        model_path,
        # Optional: Set visual token range for efficiency
        # min_pixels=256*28*28,
        # max_pixels=1280*28*28,
    )
    print("Processor loaded successfully.")
except Exception as e:
    raise Exception(f"Failed to load processor: {e}")

# Define messages with image input
# Example: Use a local image file or URL
image_path = "D:\Workspace\OpenLLM\photos\monitor_screen\\01-11-2023 10.35.18.jpg"  # Replace with your image path or URL (e.g., "https://example.com/image.jpg")
if not os.path.exists(image_path) and not image_path.startswith("http"):
    raise FileNotFoundError(f"Image file not found: {image_path}")

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": image_path,  # Local path or URL to the image
            },
            {"type": "text", "text": "Extract the vital sign BP, HR, RR, and SpO2 from the monitor screen."},
        ],
    }
]

# Prepare inputs for inference
try:
    # Apply chat template
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    # Process vision inputs
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    # Move inputs to the appropriate device
    inputs = inputs.to(device)
except Exception as e:
    raise Exception(f"Failed to prepare inputs: {e}")

# Generate output
try:
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print("Generated output:", output_text)
except Exception as e:
    raise Exception(f"Failed during inference: {e}")