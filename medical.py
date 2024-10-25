import os
import multiprocessing as mp
import numpy as np
import pandas as pd
import tiktoken
from tqdm import tqdm

# Configuration
class Config:
    INPUT_DATA_DIR = r"/content/drive/MyDrive/Medical_LLM/input_data"
    LOCAL_DIR = "/content/drive/MyDrive/Medical_LLM/medical_dataset_cache"
    SHARD_SIZE = int(1e3)  # tokens per shard

def load_datasets(input_data_dir):
    """Load and concatenate all CSV files from the input directory."""
    dataframes = []
    for filename in os.listdir(input_data_dir):
        if filename.endswith('.csv'):
            filepath = os.path.join(input_data_dir, filename)
            df = pd.read_csv(filepath)
            dataframes.append(df)
            print(f'File loaded: {filepath}')
    return pd.concat(dataframes, ignore_index=True)

class TokenProcessor:
    def __init__(self):
        self.enc = tiktoken.get_encoding("gpt2")
        self.eot = self.enc._special_tokens['<|endoftext|>']
        self.global_unique_tokens = set()
    
    def tokenize(self, doc):
        """Tokenize a single document."""
        text = f"Q: {doc['Question']} A: {doc['Answer']}"
        tokens = [self.eot]
        tokens.extend(self.enc.encode_ordinary(text))
        self.global_unique_tokens.update(tokens)
        
        tokens_np = np.array(tokens)
        assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "token dictionary too large for uint16"
        return tokens_np.astype(np.uint16)

class ShardWriter:
    def __init__(self, cache_dir, shard_size):
        self.cache_dir = cache_dir
        self.shard_size = shard_size
        self.current_shard = np.empty((shard_size,), dtype=np.uint16)
        self.token_count = 0
        self.shard_index = 0
        self.progress_bar = None
        
    def write_shard(self, tokens=None, final=False):
        """Write the current shard to disk."""
        if final:
            if self.token_count == 0:
                return
            tokens_to_write = self.current_shard[:self.token_count]
        else:
            tokens_to_write = self.current_shard
            
        split = "val" if self.shard_index == 0 else "train"
        filename = os.path.join(self.cache_dir, f"medical_{split}_{self.shard_index:06d}")
        np.save(filename, tokens_to_write)
        
        if self.progress_bar:
            self.progress_bar.close()
        
        self.shard_index += 1
        self.token_count = 0
        self.progress_bar = None
    
    def add_tokens(self, tokens):
        """Add tokens to the current shard, writing to disk if full."""
        while len(tokens) > 0:
            if self.progress_bar is None:
                self.progress_bar = tqdm(total=self.shard_size, 
                                       unit="tokens", 
                                       desc=f"Shard {self.shard_index}")
            
            space_left = self.shard_size - self.token_count
            tokens_to_add = tokens[:space_left]
            self.current_shard[self.token_count:self.token_count + len(tokens_to_add)] = tokens_to_add
            self.token_count += len(tokens_to_add)
            self.progress_bar.update(len(tokens_to_add))
            
            if self.token_count == self.shard_size:
                self.write_shard()
                
            tokens = tokens[len(tokens_to_add):]

def main():
    # Create cache directory
    os.makedirs(Config.LOCAL_DIR, exist_ok=True)
    
    # Load dataset
    fw = load_datasets(Config.INPUT_DATA_DIR)
    print(f"Dataset shape: {fw.shape}")
    print(f"Dataset info:\n{fw.info()}")
    
    # Initialize processors
    token_processor = TokenProcessor()
    shard_writer = ShardWriter(Config.LOCAL_DIR, Config.SHARD_SIZE)
    
    # Process documents
    nprocs = max(1, os.cpu_count() // 2)
    with mp.Pool(nprocs) as pool:
        for tokens in pool.imap(token_processor.tokenize, 
                              fw.to_dict(orient='records'), 
                              chunksize=16):
            shard_writer.add_tokens(tokens)
    
    # Write final shard
    shard_writer.write_shard(final=True)
    
    print(f"Final Global Vocabulary Size: {len(token_processor.global_unique_tokens)}")
    print(f"Total shards created: {shard_writer.shard_index}")

if __name__ == '__main__':
    main()