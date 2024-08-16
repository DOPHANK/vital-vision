from transformers import LlamaForCausalLM, LlamaTokenizer

# Specify the model name or path
model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"

# Re-attempt to download and load the tokenizer and model
try:
    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    model = LlamaForCausalLM.from_pretrained(model_name, device_map="auto")
except OSError as e:
    print(f"Error loading model or tokenizer: {e}")
    # Try removing any existing directory with the same name if it exists locally
    import shutil
    shutil.rmtree(model_name, ignore_errors=True)
    # Retry download
    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    model = LlamaForCausalLM.from_pretrained(model_name, device_map="auto")

# Save the model locally for future use
model.save_pretrained("./llama_3_1_model")
tokenizer.save_pretrained("./llama_3_1_model")
