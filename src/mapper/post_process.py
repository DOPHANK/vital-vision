import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain.llms.base import LLM
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from typing import Any, Optional, List
import os 

# Load the Llama model and tokenizer
model_name = "meta-llama/Llama-3.1-8B-Instruct"
print(f"Starting to load the model {model_name} into memory")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    token = os.getenv("HUGGINGFACE_TOKEN")
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
# tokenizer.bos_token_id = 1

# Define the custom LLM class
class LlamaVitalSignsExtractor(LLM):
    model: Any
    tokenizer: Any

    def __init__(self, model, tokenizer):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer

    @property
    def _llm_type(self) -> str:
        return "llama_vital_signs_extractor"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=100,
            pad_token_id=self.tokenizer.eos_token_id  # Explicitly set pad_token_id
        )
        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        if stop is not None:
            for s in stop:
                if s in text:
                    text = text[:text.index(s)]

        return text[len(prompt):]

# Define the OCRPostProcessor class
class OCRPostProcessor:
    def __init__(self, llm=None):
        self.llm = llm if llm else LlamaVitalSignsExtractor(model, tokenizer)

    def process_text(self, ocr_text: str) -> str:
        """Process the extracted OCR text using LangChain."""
        prompt = PromptTemplate(input_variables=["ocr_text"], template="Extract vital sign data from the text: {ocr_text}")
        chain = LLMChain(llm=self.llm, prompt=prompt)

        return chain.run({"ocr_text": ocr_text})

# Example usage
def main():
    processor = OCRPostProcessor()
    test_text = "Patient vitals: BP 120/80, HR 72, Temp 98.6F, RR 16"
    result = processor.process_text(test_text)
    print(f"Extracted vital signs: {result}")

if __name__ == "__main__":
    main()
