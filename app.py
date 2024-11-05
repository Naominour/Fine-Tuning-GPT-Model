import streamlit as st
import torch
from train_gpt2 import GPT, GPTConfig, enc

# Load the trained GPT model
model_path = "path/to/saved/model.pt"
checkpoint = torch.load(model_path)
config = checkpoint['config']
model = GPT(config)
model.load_state_dict(checkpoint['model'])
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

def generate_text(prompt, num_return_sequences=4, max_length=100):
    tokens = enc.encode(prompt)
    tokens = torch.tensor([tokens], dtype=torch.long, device=device)
    generated = model.generate(
        tokens,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        num_return_sequences=num_return_sequences,
        max_length=max_length,
    )
    
    generated_text = []
    for i in range(num_return_sequences):
        sample = generated[i].tolist()
        text = enc.decode(sample)
        generated_text.append(text)
    
    return generated_text

st.title("GPT Text Generation")

prompt = st.text_input("Enter a prompt:", "Smoking is the major risk factor for")

if st.button("Generate Text"):
    generated_text = generate_text(prompt, num_return_sequences=4)
    
    for i, text in enumerate(generated_text):
        st.write(f"Sample {i+1}:")
        st.write(text)