import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
# from langchain.llms import Llama
from langchain import LLMChain, PromptTemplate
from dotenv import load_dotenv
import os


# Load environment variables from .env file
load_dotenv()

class CustomLlamaLLM:
    def __init__(self, model, tokenizer, device='cpu'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def generate(self, prompt: str) -> str:
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
    model = LlamaForCausalLM.from_pretrained(model_name, use_auth_token=token)
    
    # Load the weights from the checkpoint
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    
    # Load the tokenizer
    tokenizer = LlamaTokenizer.from_pretrained(model_name, use_auth_token=token)
    
    # Move the model to the specified device
    model.to(device)
    
    return model, tokenizer

# def integrate_with_langchain(text_input, model, tokenizer, device='cuda'):
#     """
#     Integrate the LLAMA model with LangChain to process text input.

#     Parameters:
#     -----------
#     text_input : str
#         The text input to process.
#     model : LlamaForCausalLM
#         The LLAMA model loaded from the checkpoint.
#     tokenizer : LlamaTokenizer
#         The tokenizer for the LLAMA model.
#     device : str
#         The device to run the model on ('cpu' or 'cuda').

#     Returns:
#     --------
#     output : str
#         The processed output from the LLAMA model.
#     """
#     # Create a LangChain LLM object
#     llm = Llama(model=model, tokenizer=tokenizer, device=device)
    
#     # Define a simple prompt template
#     prompt_template = PromptTemplate(
#         input_variables=["text"],
#         template="Here is the processed output: {text}"
#     )
    
#     # Create a LangChain pipeline
#     llm_chain = LLMChain(prompt_template=prompt_template, llm=llm)
    
#     # Run the input text through the pipeline
#     output = llm_chain.run({"text": text_input})
    
#     return output


def integrate_with_langchain(text_input, model, tokenizer, device='cpu'):
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

# Integrate with Langchain
if __name__ == "__main__":
    input_text = "Can CAl Thời gian Nhận định tình trạng duc_ gnh_da Wul?LQ 7e ducp Mê bel Bakh tiau 4a SQ0M0 B2C Aqaua Alaiu ZA #nÉ_qua_QL [Uloitu_zunclc BD r Ild_uba{ Juâc 4242 Ácuay 'Lat (ô4 Da qLdLADdL da JLal_claa 4ì2 à AliL [dd_chan 1@_ha Me Ju (2 ALqLA Jlaud Zc_qua ala Uel RLul IN L IBsoh_h28a ) xu4 Nan_du_ea2 22 [dhuiss Hax_Dxy 410  Llenhai Hls 22L 9P291 LQ Hôo_QeEmg qa Qiêt_ NB Annh vQ Vê Chlohcxidine 40 # That sành 43# 2Cing kucoỳ C2 30 464 442 LLLL ax1d4 Jleubyf 2ucn / Jluàadt gu421 day dÊd zqử AULl bPsDandccl Auug dgR 7v +l'ng {lulel_ Aq+4 1B Mzy Jop 112+"
    result = integrate_with_langchain(input_text)
    print(result)
    