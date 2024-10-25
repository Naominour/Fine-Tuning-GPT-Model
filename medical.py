
import os
import multiprocessing as mp
import numpy as np
import pandas as pd
import tiktoken
from datasets import load_dataset  # pip install datasets
from tqdm import tqdm  # pip install tqdm

# ------------------------------------------
input_data_dir = r"/content/drive/MyDrive/Medical_LLM/input_data"
global_unique_tokens = set()


def load_datasets(input_data_dir):
    dataframes = []
    for filename in os.listdir(input_data_dir):
        if filename.endswith('.csv'):
            filepath = os.path.join(input_data_dir, filename)
            df = pd.read_csv(filepath)
            dataframes.append(df)
            print(f'file append:' + filepath)
    return pd.concat(dataframes, ignore_index=True)


fw = load_datasets(input_data_dir)

local_dir = "/content/drive/MyDrive/Medical_LLM/medical_dataset_cache"
shard_size = int(1e6)  # 100M tokens per shard, total of 100 shards

# create the cache the local directory if it doesn't exist yet
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

# download the dataset
# fw = load_dataset("gamino/wiki_medical_terms")

# Initialize the tokenizer outside of the main function
enc = tiktoken.get_encoding("gpt2")
eot = enc.encode("<|endoftext|>", allowed_special={'<|endoftext|>'})[0] # end of text token
global_unique_tokens = set()

max_valid_token_id = 22730
replacement_token_id = eot

def tokenize(doc):
    # tokenizes a single document and returns a numpy array of uint16 tokens
    text = f"Q: {doc['Question']} A: {doc['Answer']}"

    tokens = [eot]  # the special <|endoftext|> token delimits all documents
    tokens.extend(enc.encode_ordinary(text))
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "token dictionary too large for uint16"
    tokens_np_uint16 = tokens_np.astype(np.uint16)
    return tokens_np_uint16, set(tokens)

def write_datafile(filename, tokens_np):
    np.save(filename, tokens_np)

def main():
    global global_unique_tokens

    print(f"Dataset structure: {fw.info()}")
    print(f"Dataset shape: {fw.shape}")
    print(f"Final Global Vocabulary Size: {len(global_unique_tokens)}")

    # tokenize all documents and write output shards, each of shard_size tokens (last shard has remainder)
    nprocs = max(1, os.cpu_count() // 2)
    with mp.Pool(nprocs) as pool:
        shard_index = 0
        # preallocate buffer to hold current shard
        all_tokens_np = np.empty((shard_size,), dtype=np.uint16)
        token_count = 0
        progress_bar = None
        for tokens, unique_tokens in pool.imap(tokenize, fw.to_dict(orient='records'), chunksize=16):
            global_unique_tokens.update(unique_tokens)
            # is there enough space in the current shard for the new tokens?
            if token_count + len(tokens) < shard_size:
                # simply append tokens to current shard
                all_tokens_np[token_count:token_count + len(tokens)] = tokens
                token_count += len(tokens)
                # update progress bar
                if progress_bar is None:
                    progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_index}")
                progress_bar.update(len(tokens))
            else:
                # write the current shard and start a new one
                split = "val" if shard_index == 0 else "train"
                filename = os.path.join(DATA_CACHE_DIR, f"medical_{split}_{shard_index:06d}")
                # split the document into whatever fits in this shard; the remainder goes to next one
                remainder = shard_size - token_count
                progress_bar.update(remainder)
                all_tokens_np[token_count:token_count + remainder] = tokens[:remainder]
                write_datafile(filename, all_tokens_np)
                shard_index += 1
                progress_bar = None
                # populate the next shard with the leftovers of the current doc
                all_tokens_np[0:len(tokens) - remainder] = tokens[remainder:]
                token_count = len(tokens) - remainder

        # write any remaining tokens as the last shard
        if token_count != 0:
            split = "val" if shard_index == 0 else "train"
            filename = os.path.join(DATA_CACHE_DIR, f"medical_{split}_{shard_index:06d}")
            write_datafile(filename, all_tokens_np[:token_count])
    # Adjust each shard by replacing tokens that exceed max_valid_token_id
    for shard_filename in os.listdir(local_dir):
        if shard_filename.endswith('.npy'):
            shard_path = os.path.join(local_dir, shard_filename)
            tokens = np.load(shard_path)

            # Replace tokens exceeding max_valid_token_id
            tokens = np.where(tokens > max_valid_token_id, replacement_token_id, tokens)
            np.save(shard_path, tokens)

    print("Completed replacing tokens exceeding max vocabulary size.")

    vocab_size = len(global_unique_tokens)
    print(f"Final Global Vocabulary Size: {vocab_size}")

if __name__ == '__main__':
    main()