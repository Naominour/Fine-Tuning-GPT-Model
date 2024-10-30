import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken
import streamlit as st

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size() 
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True) # flash attention
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu    = nn.GELU(approximate='tanh')
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPTConfig:
    block_size: int = 1024 # max sequence length
    vocab_size: int = 50257 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 12 # number of layers
    n_head: int = 12 # number of heads
    n_embd: int = 768 # embedding dimension


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight

        # init params
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        # idx is of shape (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # forward the token and posisition embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

@st.cache_resource
def load_model():
    config = GPTConfig()
    model = GPT(config)
    model.load_state_dict(torch.load("G:/My Drive/Medical_LLM/medical_dataset_cache/saved_model.pth", map_location=torch.device('cpu'), weights_only=True))
    model.eval()  # Set the model to evaluation mode
    return model

# Initialize tokenizer
enc = tiktoken.get_encoding("gpt2")
eot = enc.encode("<|endoftext|>", allowed_special={'<|endoftext|>'})[0]

# Function to generate text based on input
def generate_text(model, input_text, max_tokens=100, temperature=0.7):
    # Tokenize input
    input_ids = enc.encode(input_text)
    input_ids = torch.tensor(input_ids).unsqueeze(0)  # Add batch dimension
    
    generated_tokens = []
    
    with torch.no_grad():
        for _ in range(max_tokens):
            # Get model predictions
            outputs = model(input_ids)
            next_token_logits = outputs[0, -1, :] / temperature
            
            # Apply softmax to get probabilities
            probs = F.softmax(next_token_logits, dim=-1)
            
            # Sample from the distribution
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Break if we generate EOT token
            if next_token.item() == eot:
                break
                
            # Append to generated tokens
            generated_tokens.append(next_token.item())
            
            # Update input_ids with the new token
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
    
    # Decode generated tokens
    generated_text = enc.decode(generated_tokens)
    return input_text + generated_text

# Streamlit Interface
st.title("Medical Text Generation Model")
st.write("Enter the beginning of a sentence, and the model will generate a continuation.")

# Add temperature slider
temperature = st.slider("Temperature (higher = more creative, lower = more focused)", 0.1, 1.0, 0.7, 0.1)

# Add max tokens slider
max_tokens = st.slider("Maximum tokens to generate", 10, 500, 100, 10)

# Use st.text_input for user to enter their own sentence
input_text = st.text_input("Input text:", "")

# Generate text when the button is clicked
if st.button("Generate"):
    if input_text:  # Ensure input_text is not empty
        with st.spinner("Generating..."):
            generated_text = generate_text(model, input_text, max_tokens=max_tokens, temperature=temperature)
        st.write("Generated Text:")
        st.write(generated_text)
    else:
        st.write("Please enter some text to begin generation.")