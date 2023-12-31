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
        num_return_sequences=10,  # Generates 3 sequences
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
prompt = st.text_input("Enter a prompt to get started:", "To be, or not to be, that is the question")

# Image of Shakespeare bust
image_html = """
<div style="text-align:center">
    <img src="https://images.unsplash.com/photo-1581344895000-b5deedbd1660?q=80&w=300&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D" alt="Shakespeare" style="width:200px;">
</div>
"""
st.sidebar.markdown(image_html, unsafe_allow_html=True)

# HTML caption for the image
image_caption = """
Photo by <a href="https://unsplash.com/@birminghammuseumstrust?utm_content=creditCopyText&utm_medium=referral&utm_source=unsplash">Birmingham Museums Trust</a> on <a href="https://unsplash.com/photos/white-ceramic-man-head-bust-L2sbcLBJwOc?utm_content=creditCopyText&utm_medium=referral&utm_source=unsplash">Unsplash</a>
"""
st.sidebar.markdown(image_caption, unsafe_allow_html=True)

# Sidebar for About section
with open('about.txt', 'r') as file:
    about_content = file.read()
st.sidebar.markdown(about_content)

if st.button('Generate Text'):
    if prompt:
        with st.spinner('Generating...'):
            generated_text = generate_shakespeare_text(model, tokenizer, prompt)
        st.write(generated_text)