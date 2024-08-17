from transformers import LlamaForCausalLM, LlamaTokenizer
import torch
from transformers import LlamaTokenizer

# Download and save the tokenizer from Hugging Face
# tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B")
# tokenizer.save_pretrained("/media/data/Workspace/vital-ocr/model_checkpoints/Meta-Llama-3.1-8B/original/")


# model_dir = "/media/data/Workspace/vital-ocr/model_checkpoints/Meta-Llama-3.1-8B/original/"
# # Load the tokenizer
# tokenizer = LlamaTokenizer.from_pretrained(model_dir)

# # Load the model
# model = LlamaForCausalLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16)

from transformers import LlamaTokenizer

# Specify the model directory where you want to save the tokenizer
model_dir = "/media/data/Workspace/vital-ocr/model_checkpoints/Meta-Llama-3.1-8B/redownload/"

# Download and save the tokenizer
tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B")
tokenizer.save_pretrained(model_dir)
