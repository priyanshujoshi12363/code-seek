import torch
import os
import sys
sys.path.insert(0, '.')
from model.transformer import CodeSeek
from config import *
from tokenizers import Tokenizer

def load_model(path):
    model = CodeSeek(VOCAB_SIZE, D_MODEL, N_HEADS, N_LAYERS, FFN_DIM, MAX_SEQ_LEN, DROPOUT)
    ckpt = torch.load(path, map_location='cpu')
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model.load_state_dict(ckpt)
    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    return model

def clean_text(text):
    text = text.replace('Ġ', ' ')
    text = text.replace('Ċ', '\n')
    text = text.replace('ĉ', '')
    text = ' '.join(text.split())
    return text.strip()

def chat():
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    
    model_path = f'{CHECKPOINT_DIR}/step_15000.pt'
    if not os.path.exists(model_path):
        model_path = f'{CHECKPOINT_DIR}/best_model.pt'
    if not os.path.exists(model_path):
        print(f"Model not found")
        return
    
    model = load_model(model_path)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("CodeSeek ready. Type 'quit' to exit.\n")
    
    while True:
        user = input("You: ")
        if user.lower() == 'quit':
            break
        
        tokens = tokenizer.encode(user).ids
        tokens = torch.tensor([tokens], dtype=torch.long).to(device)
        
        output = model.generate(tokens, max_new_tokens=150, temperature=0.8, top_k=50)
        output_tokens = output[0].cpu().tolist()
        response = tokenizer.decode(output_tokens)
        response = clean_text(response)
        
        print(f"Bot: {response}\n")

if __name__ == "__main__":
    chat()