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
    text = text.replace('U ser :', '')
    text = text.replace('A ss ist ant :', '')
    text = text.replace('Bot :', '')
    text = text.replace('User:', '')
    text = text.replace('Assistant:', '')
    text = text.replace('Bot:', '')
    text = text.replace('Ġ', ' ')
    text = text.replace('Ċ', '\n')
    text = ' '.join(text.split())
    return text.strip()

def chat():
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    
    model_paths = [
        'checkpoints/knoc8-Chat-v3.pt',
        'checkpoints/knoc8_chat_v3/epoch_5.pt',
        'checkpoints/knoc8_chat_v3/best_model.pt',
        'checkpoints/knoc8.pt',
    ]
    
    model_path = None
    for path in model_paths:
        if os.path.exists(path):
            model_path = path
            break
    
    if model_path is None:
        print("No model found!")
        return
    
    print(f"Loaded: {model_path}")
    model = load_model(model_path)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("\n" + "=" * 50)
    print("knoc8 - Your AI Storyteller & Chat Assistant")
    print("Type 'quit' to exit")
    print("=" * 50 + "\n")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            print("\nknoc8: Catch you later! Come back for more stories! ✨")
            break
        
        prompt = f"User: {user_input}\nAssistant:"
        tokens = tokenizer.encode(prompt).ids
        tokens = torch.tensor([tokens], dtype=torch.long).to(device)
        
        output = model.generate(tokens, max_new_tokens=200, temperature=0.8, top_k=50)
        output_tokens = output[0].cpu().tolist()
        response = tokenizer.decode(output_tokens)
        response = clean_text(response)
        
        if not response or len(response) < 5:
            response = "Let me think of a good story for you! What kind would you like?"
        
        print(f"knoc8: {response}\n")

if __name__ == "__main__":
    chat()