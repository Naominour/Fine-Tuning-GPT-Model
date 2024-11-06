import streamlit as st
import torch
import tiktoken
import torch.nn.functional as F
import sys
import os

# Add the directory containing train_gpt2.py to Python path
# Adjust this path to where your train_gpt2.py is located
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the model architecture from your training script
from train_gpt2 import GPT, GPTConfig

@st.cache_resource
def load_model(model_path):
    """Load the saved model"""
    try:
        # Initialize model with the same configuration
        model = GPT(GPTConfig())
        
        # Load the saved state dictionary
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        
        # If the checkpoint is a dictionary containing both model and config
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()  # Set to evaluation mode
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def generate_text(model, device, prompt, num_samples=4):
    """Generate text using the model"""
    model.eval()
    max_length = 64
    
    # Initialize tokenizer
    enc = tiktoken.get_encoding("gpt2")
    
    # Tokenize input prompt
    tokens = enc.encode(prompt)
    tokens = torch.tensor(tokens, dtype=torch.long)
    tokens = tokens.unsqueeze(0).repeat(num_samples, 1)
    xgen = tokens.to(device)
    
    # Set up random generator
    sample_rng = torch.Generator(device=device)
    sample_rng.manual_seed(42)
    
    with torch.no_grad():
        while xgen.size(1) < max_length:
            # Forward pass
            logits, _ = model(xgen)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            
            # Top-k sampling
            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
            ix = torch.multinomial(topk_probs, 1, generator=sample_rng)
            xcol = torch.gather(topk_indices, -1, ix)
            
            # Append to sequence
            xgen = torch.cat((xgen, xcol), dim=1)
    
    # Decode generated sequences
    generated_text = []
    for i in range(num_samples):
        tokens = xgen[i, :max_length].tolist()
        decoded = enc.decode(tokens)
        generated_text.append(decoded)
    
    return generated_text

def main():
    st.title("Medical Text Generation")
    st.write("Enter your medical prompt and get AI-generated completions")
    
    # Device selection
    device = "cuda" if torch.cuda.is_available() else "cpu"
    st.info(f"Using device: {device}")
    
    # Model loading
    if 'model' not in st.session_state:
        with st.spinner("Loading model..."):
            model_path = "G:/My Drive/Medical_LLM/medical_dataset_cache/saved_model.pth"
            st.session_state.model = load_model(model_path)
            if st.session_state.model is not None:
                st.session_state.model.to(device)
                st.success("Model loaded successfully!")
            else:
                st.error("Failed to load model.")
                return
    
    # Create input area
    user_input = st.text_area(
        "Enter your prompt:",
        value="Smoking is the major risk factor for",
        help="Type your medical prompt here"
    )
    
    # Number of samples selector
    num_samples = st.slider("Number of samples", 1, 4, 4)
    
    # Generate button
    if st.button("Generate"):
        if user_input:
            try:
                with st.spinner("Generating text..."):
                    generations = generate_text(
                        st.session_state.model,
                        device,
                        user_input,
                        num_samples
                    )
                
                # Display generations
                st.subheader("Generated Texts:")
                for i, text in enumerate(generations):
                    with st.container():
                        st.markdown(f"**Sample {i}:**")
                        st.write(text)
                        st.divider()
                        
            except Exception as e:
                st.error(f"Error during generation: {str(e)}")
                st.error("Full error message:", exc_info=True)
        else:
            st.warning("Please enter a prompt first.")

if __name__ == "__main__":
    main()