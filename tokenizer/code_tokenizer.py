# tokenizer/code_tokenizer.py
import glob
from tokenizers import Tokenizer
import torch
import os

def tokenize_in_chunks(file_path, tokenizer, max_seq, chunk_size=1000000):
    """Tokenize a large file in smaller pieces"""
    chunks = []
    
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        buffer = ""
        file_pos = 0
        
        while True:
            piece = f.read(chunk_size)
            if not piece:
                break
            
            buffer += piece
            file_pos += len(piece)
            
            # Tokenize when buffer is full
            if len(buffer) >= chunk_size * 2:
                tokens = tokenizer.encode(buffer).ids
                for i in range(0, len(tokens) - max_seq, max_seq):
                    chunks.append(tokens[i:i + max_seq + 1])
                buffer = ""  # Clear buffer
                
                print(f"  Progress: {file_pos/(1024*1024):.0f} MB processed, {len(chunks):,} chunks")
        
        # Process remaining buffer
        if buffer:
            tokens = tokenizer.encode(buffer).ids
            for i in range(0, len(tokens) - max_seq, max_seq):
                chunks.append(tokens[i:i + max_seq + 1])
    
    return chunks

def tokenize_code_data():
    tokenizer = Tokenizer.from_file("tokenizer/codeseek_tokenizer.json")
    max_seq = 256
    all_chunks = []
    
    code_files = glob.glob("data/code_data/*.txt")
    print(f"Found {len(code_files)} code files:")
    for f in code_files:
        size_mb = os.path.getsize(f) / (1024 * 1024)
        print(f"  - {os.path.basename(f)} ({size_mb:.1f} MB)")
    
    print("\nTokenizing in chunks...")
    
    for f in code_files:
        print(f"\nProcessing: {os.path.basename(f)}")
        chunks = tokenize_in_chunks(f, tokenizer, max_seq)
        all_chunks.extend(chunks)
        print(f"  Done! Total chunks: {len(all_chunks):,}")
    
    save_path = "data/code_data/tokenized_code.pt"
    torch.save(all_chunks, save_path)
    
    file_size_mb = os.path.getsize(save_path) / (1024 * 1024)
    print(f"\nSaved {len(all_chunks):,} chunks ({file_size_mb:.1f} MB)")
    print(f"   Path: {save_path}")

if __name__ == "__main__":
    tokenize_code_data()