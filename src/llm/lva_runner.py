import os
import torch
from PIL import Image
from huggingface_hub import snapshot_download
from llava.model.builder import load_pretrained_model
from llava.conversation import conv_templates
from llava.mm_utils import process_images
from llava.constants import DEFAULT_IMAGE_TOKEN
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def tokenizer_image_token_patched(prompt, tokenizer, image_token_index, return_tensors=None):
    logger.info("Tokenizing prompt: %s", prompt)
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split('<image>')]
    if not prompt_chunks or all(len(chunk) == 0 for chunk in prompt_chunks):
        raise ValueError("Prompt is empty or contains no valid tokens after splitting on <image>")

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    if not input_ids:
        raise ValueError("Generated input_ids is empty after processing prompt")

    logger.info("Generated input_ids: %s", input_ids)
    if return_tensors == 'pt':
        return torch.tensor(input_ids, dtype=torch.long)
    return input_ids

def download_model_if_missing(model_path, hf_repo):
    if not os.path.exists(model_path):
        logger.info(f"Downloading model from HuggingFace: {hf_repo}")
        snapshot_download(repo_id=hf_repo, local_dir=model_path, local_dir_use_symlinks=False, force_download=True)
        logger.info(f"Model downloaded to: {model_path}")
    else:
        logger.info(f"Model found: {model_path}")

def run_llava_pipeline(model_path, image_path, question, hf_repo, device='cuda' if torch.cuda.is_available() else 'cpu'):
    try:
        download_model_if_missing(model_path, hf_repo)

        logger.info("Loading model...")
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path=model_path,
            model_base=None,
            model_name=os.path.basename(model_path),
            device=device
        )
        logger.info("Model loaded: %s", model is not None)
        logger.info("Tokenizer: %s", tokenizer)

        # Add <image> to tokenizer
        logger.info("Adding <image> to tokenizer if missing")
        if DEFAULT_IMAGE_TOKEN not in tokenizer.get_vocab():
            tokenizer.add_special_tokens({'additional_special_tokens': [DEFAULT_IMAGE_TOKEN]})
            tokenizer.vocab[DEFAULT_IMAGE_TOKEN] = 32000  # Force ID 32000
            logger.info("Added <image> token with ID 32000")

        # Optimize for CPU
        if device == 'cpu':
            logger.info("Setting model to float16 for CPU")
            model = model.to(torch.float16)
            if hasattr(model.get_model(), "initialize_vision_tower"):
                model.get_model().initialize_vision_tower()
            vision_tower = model.get_model().vision_tower
            logger.info("Vision tower: %s", vision_tower)
            if vision_tower is None:
                raise RuntimeError("Vision tower initialization failed!")
            vision_tower.to(device=torch.device("cpu"), dtype=torch.float16)
            if hasattr(vision_tower, 'model'):
                vision_tower.model.to(device=torch.device("cpu"), dtype=torch.float16)
            mm_projector = getattr(model.get_model(), 'mm_projector', None)
            if mm_projector is not None:
                mm_projector.to(device=torch.device("cpu"), dtype=torch.float16)

        # Load and preprocess image
        logger.info("Loading image: %s", image_path)
        image = Image.open(image_path).convert("RGB")
        image_tensor = process_images([image], image_processor, model.config).to(device)
        logger.info("Image tensor device: %s", image_tensor.device)

        # Ensure image tensor matches model dtype
        vision_dtype = model.get_model().mm_projector[0].weight.dtype
        image_tensor = image_tensor.to(dtype=vision_dtype)

        # Build conversation
        question_with_image = f"{DEFAULT_IMAGE_TOKEN} {question}"
        logger.info("Available conv templates: %s", list(conv_templates.keys()))
        template_name = "llava_v1"
        if template_name not in conv_templates:
            template_name = list(conv_templates.keys())[0]
        logger.info("Using conversation template: %s", template_name)
        conv = conv_templates[template_name].copy()
        conv.append_message(conv.roles[0], question_with_image)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        logger.info("Prompt: %s", prompt)
        logger.info("Tokenizer vocab contains DEFAULT_IMAGE_TOKEN: %s", DEFAULT_IMAGE_TOKEN in tokenizer.get_vocab())
        logger.info("Tokenized DEFAULT_IMAGE_TOKEN: %s", tokenizer.encode(DEFAULT_IMAGE_TOKEN, add_special_tokens=False))

        # Tokenize
        image_token_index = tokenizer.convert_tokens_to_ids(DEFAULT_IMAGE_TOKEN)
        logger.info("Image token index: %s", image_token_index)
        if image_token_index != 32000:
            logger.warning("Image token index is not 32000, forcing to 32000")
            image_token_index = 32000
        tokenized_output = tokenizer_image_token_patched(prompt, tokenizer, image_token_index=image_token_index, return_tensors="pt")
        logger.info("Tokenized output: %s", tokenized_output)
        input_ids = tokenized_output.unsqueeze(0).to(device) if tokenized_output is not None else None
        logger.info("Input IDs: %s", input_ids)
        logger.info("Input IDs device: %s", input_ids.device if input_ids is not None else "None")
        if input_ids is None:
            raise ValueError("Input IDs is None before model.generate")

        # Generate answer
        logger.info("Starting inference...")
        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_ids,
                images=image_tensor,
                do_sample=True,
                temperature=0.2,
                max_new_tokens=128
            )
        logger.info("Inference complete")

        # Decode output
        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        logger.info("Answer: %s", output_text.split("###")[-1].strip())
        
    except ImportError as e:
        logger.error(f"Import error: {e}")
        logger.info("Please install the required dependencies: pip install protobuf")
        raise
    except Exception as e:
        logger.error(f"Error during inference: {e}")
        raise

if __name__ == "__main__":
    model_dir = "checkpoints/liuhaotian/llava-1.5-7b"  # Or "checkpoints/liuhaotian/llava-1.5-7b"
    hf_repo = "liuhaotian/llava-1.5-7b"  # Or "liuhaotian/llava-1.5-7b"
    # model_dir = "checkpoints/llava-hf/llava-1.5-7b-hf"  # Or "checkpoints/liuhaotian/llava-1.5-7b"
    # hf_repo = "llava-hf/llava-1.5-7b-hf"  # Or "liuhaotian/llava-1.5-7b"
    image_path = "data/01-11-2023 10.28.56.jpg"
    question = "What is happening in the image?"
    run_llava_pipeline(model_dir, image_path, question, hf_repo)
    print("Finished!")
