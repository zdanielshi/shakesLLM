import streamlit as st
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import re

def generate_shakespeare_text(model, tokenizer, prompt):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    device = model.device
    input_ids = input_ids.to(device)

    output = model.generate(
        input_ids, 
        max_length=50, 
        num_return_sequences=3,  # Generates 3 sequences
        do_sample=True, 
        top_k=50
    )

    # Formatting the output using regex
    generated_texts = [tokenizer.decode(generated_seq, skip_special_tokens=True) for generated_seq in output]
    formatted_output = "\n\n---\n\n".join(generated_texts)  # Initial joining of outputs

    # Regex to start each sentence on a new line
    formatted_output = re.sub(r'(?<=[.!?])\s', r'\n', formatted_output)

    return formatted_output

# Load the model and tokenizer
model_path = "./tiny_shakespeare_gpt2"  # Update this path if needed
model = GPT2LMHeadModel.from_pretrained(model_path)
tokenizer = GPT2Tokenizer.from_pretrained(model_path)

# Streamlit interface
st.title('LLM Shakespeare Text Generator')
prompt = st.text_input("Enter your text prompt:")

if prompt:
    with st.spinner('Generating...'):
        generated_text = generate_shakespeare_text(model, tokenizer, prompt)
    st.write(generated_text)
