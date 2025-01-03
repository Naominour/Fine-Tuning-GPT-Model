import os
import math
import time
import mlflow
import inspect
import torch
import tiktoken
import numpy as np
import torch.nn as nn
import torch.distributed as dist

from dataclasses import dataclass
from torch.nn import functional as F
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import GPT2LMHeadModel
# -----------------------------------------------------------------------------

data_root = "/content/drive/MyDrive/Medical_LLM/medical_dataset_cache"

class CausalSelfAttention(nn.Module):
    """
    Implements causal self-attention for a transformer block.
    This module includes query, key, and value projections, and computes
    scaled dot-product attention with causal masking.
    """
    
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

    """
    Implements a feedforward neural network (MLP) for a transformer block.
    Consists of two linear layers and a GELU activation.
    """
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

    """
    Implements a single transformer block, consisting of self-attention and an MLP, with residual connections.
    """
    
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

@dataclass
class GPTConfig:

    """
    Configuration class for the GPT model.
    Contains hyperparameters such as block size, number of layers, and embedding dimensions.
    """
    block_size: int = 1024 # max sequence length
    vocab_size: int = 50257 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 12 # number of layers
    n_head: int = 12 # number of heads
    n_embd: int = 768 # embedding dimension

class GPT(nn.Module):

    """
    Implements the GPT model using a stack of transformer blocks.
    Includes token embeddings, positional embeddings, and an output layer.
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd), # Token embeddings
            wpe = nn.Embedding(config.block_size, config.n_embd), # Positional embeddings
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]), # Stack of transformer blocks
            ln_f = nn.LayerNorm(config.n_embd), # Final layer normalisation
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False) # Output layer (logits)

        # Weight tying between input embedding and output layer
        self.transformer.wte.weight = self.lm_head.weight

        # Initialise model parameters
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5 # Scale initialisation for deeper layers
            torch.nn.init.normal_(module.weight, mean=0.0, std=std) # Initialise weights with normal distribution
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias) # Initialise biases to zero
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02) # Initialise embedding weights

    def forward(self, idx, targets=None):
        # idx is of shape (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # Token and positional embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)  # Positional indices
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb  # Combine token and positional embeddings
        # Pass through transformer blocks
        for block in self.transformer.h:
            x = block(x)
            
        # Final layer normalisation and output projection
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # Compute logits for vocabulary (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        print("loading weights from pretrained gpt: %s" % model_type)

        # Set configuration based on the selected model type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # Set vocabulary size
        config_args['block_size'] = 1024 # Set maximum block size
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        # Load pretrained weights
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # Ignore attention bias

        # Load Hugging Face model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # Map weights from Hugging Face model to custom GPT model
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        
        # Transpose weights for Conv1D modules
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # Directly copy weights for other modules
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        # Configure and initialize the optimiser for training the model.
        # Collect all parameters that require gradients
        param_dict = {pn: p for pn, p in self.named_parameters()} 
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad} 
        
        # Separate parameters into two groups:
        # 1. Parameters that will be weight-decayed (e.g., 2D weight tensors in matmuls and embeddings)
        # 2. Parameters that will not be weight-decayed (e.g., biases and layer normalization weights)

        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]  # Typically, weights
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2] # Typically, biases and layernorms
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay}, # Apply weight decay to this group
            {'params': nodecay_params, 'weight_decay': 0.0}  # No weight decay for this group
        ]
        # Calculate the total number of parameters in each group for logging/debugging
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        # Log parameter counts if running on the master process
        if master_process:
            print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
            print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Check if the fused version of AdamW is available and use it if running on CUDA
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        if master_process:
            print(f"using fused AdamW: {use_fused}")
        # Initialize the AdamW optimizer with the configured parameter groups
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer

# -----------------------------------------------------------------------------
# Load tokens from the specified file
def load_tokens(filename):
    npt = np.load(filename)
    npt = npt.astype(np.int32)
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

# DataLoaderLite is a simplified loader for managing training and validation data
class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {'train', 'val'}

        # Collect and sort data shard filenames for the selected split
        data_root = "/content/drive/MyDrive/Medical_LLM/medical_dataset_cache"
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        if master_process:
            print(f"found {len(shards)} shards for split {split}")
        self.reset()

    def reset(self):
        # Start processing from the first shard
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        # Fetch a batch of data
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = (buf[:-1]).view(B, T) # # Input tokens
        y = (buf[1:]).view(B, T) # Target tokens
        # advance the position in the tensor
        self.current_position += B * T * self.num_processes
        # Advance the pointer and load the next shard if necessary
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank
        return x, y

# -----------------------------------------------------------------------------
def generate_text(model, device, device_type, ddp_rank, phrase, num_return_sequences):
    # Generates text using the trained GPT model.
    result = 'result.txt'
    model.eval() # Set model to evaluation mode
    max_length = 64 # Maximum length of the generated sequence
    tokens = enc.encode(phrase) # Encode the input phrase into tokens
    tokens = torch.tensor(tokens, dtype=torch.long)
    tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
    xgen = tokens.to(device) # Input tokens for generation
    sample_rng = torch.Generator(device=device)
    sample_rng.manual_seed(42 + ddp_rank)

    generated_text = []  # List to store generated text

    while xgen.size(1) < max_length:
        # Generate logits using the model and perform sampling
        with torch.no_grad():
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                logits, loss = model(xgen) # Forward pass to get logits (B, T, vocab_size)
            # take the logits at the last position
            logits = logits[:, -1, :] # (B, vocab_size)
            # Convert logits to probabilities
            probs = F.softmax(logits, dim=-1)
            # Top-k sampling
            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1) # return top k high probability
            # select a token from the top-k probabilities
            ix = torch.multinomial(topk_probs, 1, generator=sample_rng) # (B, 1)
            # gather the corresponding indices
            xcol = torch.gather(topk_indices, -1, ix) # Retrieve sampled token index , (B, 1)
            # append to the sequence
            xgen = torch.cat((xgen, xcol), dim=1)

    # Decode generated tokens and save the text
    for i in range(num_return_sequences):
        tokens = xgen[i, :max_length].tolist() # Truncate to max length
        decoded = enc.decode(tokens) # Decode tokens to text
        print(f"rank {ddp_rank} sample {i}: {decoded}")
        generated_text.append(decoded)

    # Save the model state to a file
    torch.save(model.state_dict(), data_root + '/saved_model.pth')

    # Save the generated medical text to a file
    output_file = data_root + "/" + result
    with open(output_file, "a") as out_f:
        for i, med in enumerate(generated_text):
            out_f.write(f"sample {i}: {med}\n")

    return generated_text  # Return the list of generated sequences

# --------------------------------------------------------------------------
# Set up Distributed Data Parallel (DDP) environment or single-process setup
ddp = int(os.environ.get('RANK', -1)) != -1  # Check if DDP is enabled
if ddp:
    # use of DDP atm demands CUDA, we set the device appropriately according to rank
    assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
else:
    # vanilla, non-DDP run
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    # Auto-detect the device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}")

# Determine device type
device_type = "cuda" if device.startswith("cuda") else "cpu"

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

enc = tiktoken.get_encoding("gpt2")

# Set batch sizes and gradient accumulation steps
total_batch_size = 16384 #  Total batch size in number of tokens
B = 16 # Micro batch size
T = 1024 # Sequence length
assert total_batch_size % (B * T * ddp_world_size) == 0, "make sure total_batch_size is divisible by B * T * ddp_world_size"
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
if master_process:
    print(f"total desired batch size: {total_batch_size}")
    print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

# Initialize data loaders for training and validation
train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="train")
val_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="val")

print("data is ready....")

torch.set_float32_matmul_precision('high') # Enable high-precision matrix multiplication

# Initialize model
model = GPT(GPTConfig(vocab_size=50257))
model.to(device)
use_compile = False # torch.compile interferes with HellaSwag eval and Generation. 
if use_compile:
    model = torch.compile(model)
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank]) # Wrap model for DDP
raw_model = model.module if ddp else model # Unwrap raw model for single process

# Define learning rate schedule
max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 715
max_steps = 19073 # Total number of steps for training (~1 epoch for 10B tokens and batch size 0.5M)
def get_lr(it):
    # Calculates the learning rate for the current training step.
    # 1) Linear warmup for the first `warmup_steps`
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    # 2) If the step exceeds the maximum steps, use the minimum learning rate
    if it > max_steps:
        return min_lr
    # 3) Apply cosine decay between `warmup_steps` and `max_steps`
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps) # Ratio of progress after warmup
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # Cosine decay coefficient
    return min_lr + coeff * (max_lr - min_lr) # Interpolated learning rate

# Initialize the optimizer
optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device_type=device_type)

# Create the directory for logging and saving checkpoints
log_dir = data_root + "/log"
os.makedirs(log_dir, exist_ok=True)
# Create and initialize the log file
log_file = os.path.join(log_dir, f"log.txt")
with open(log_file, "w") as f: # open for writing to clear the file
    pass
    
# Start an MLflow run for logging parameters and metrics
mlflow.start_run()

# Log training parameters to MLflow
mlflow.log_param("B", B)
mlflow.log_param("total_batch_size", total_batch_size)
mlflow.log_param("max_lr", max_lr)
mlflow.log_param("min_lr", min_lr)
mlflow.log_param("warmup_steps", warmup_steps)
mlflow.log_param("max_steps", max_steps)

# Training loop
for step in range(max_steps):
    t0 = time.time() # Start time for step
    last_step = (step == max_steps - 1) # Check if this is the last training step

    # Evaluate validation loss every 250 steps or on the last step
    if step % 250 == 0 or last_step:
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 20
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(x, y)
                loss = loss / val_loss_steps # Normalize loss over validation steps
                val_loss_accum += loss.detach()
                
        # Aggregate validation loss across processes in DDP
        if ddp:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
        if master_process:
            # Log and print validation loss
            print(f"validation loss: {val_loss_accum.item():.4f}")

            mlflow.log_metric("Validation Loss:", val_loss_accum.item(), step=step)

            with open(log_file, "a") as f:
                f.write(f"{step} val {val_loss_accum.item():.4f}\n")
                
            # Save checkpoints at regular intervals or on the last step
            if step > 0 and (step % 5000 == 0 or last_step):
                # optionally write model checkpoints
                checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'config': raw_model.config,
                    'step': step,
                    'val_loss': val_loss_accum.item()
                }
                torch.save(checkpoint, checkpoint_path)

    # Generate text samples at regular intervals or on the last step
    if ((step > 0 and step % 250 == 0) or last_step) and (not use_compile):
        phrase = "Smoking is the major risk factor for"
        generate_text(model, device, device_type, ddp_rank, phrase, 4)

    print("start training the model")
    model.train()
    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        print(f"step {micro_step} of training")
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        
        if ddp:
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
            
        # Compute logits and loss
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            logits, loss = model(x, y)
        loss = loss / grad_accum_steps # Scale loss for gradient accumulation
        loss_accum += loss.detach() # Accumulate the scaled loss
        loss.backward() # Backpropagate

    # Aggregate training loss across processes in DDP
    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

    # Clip gradients to avoid exploding gradients
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    # Update learning rate for the current step
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step() # Perform optimization step
    
    # Synchronize CUDA device to ensure all computations are complete
    if device_type == "cuda":
        torch.cuda.synchronize() 
    # Log training metrics
    t1 = time.time()
    dt = t1 - t0 # time difference in seconds
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
    tokens_per_sec = tokens_processed / dt
    if master_process:
        print(f"step {step:5d} | loss: {loss_accum.item():.6f} | lr {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")
        mlflow.log_metric("Train Loss", loss_accum.item(), step=step)

        # Save training loss to log file
        with open(log_file, "a") as f:
            f.write(f"{step} train {loss_accum.item():.6f}\n")

# Save the final trained model to MLflow
mlflow.pytorch.log_model(raw_model, "model")
mlflow.end_run()

if ddp:
    destroy_process_group()

