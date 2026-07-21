import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tokenizers import Tokenizer
import os
import sys
import math

sys.path.insert(0, '.')
from model.transformer import CodeSeek
from config import *

class ChatDataset(Dataset):
    def __init__(self, file_path, tokenizer_path, max_seq_len):
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.max_seq_len = max_seq_len
        self.examples = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        blocks = text.strip().split('\n\n')
        
        for block in blocks:
            block = block.strip()
            if not block:
                continue
            
            if 'User:' in block and 'Assistant:' in block:
                tokens = self.tokenizer.encode(block).ids
                if len(tokens) > 10:
                    if len(tokens) > max_seq_len:
                        tokens = tokens[:max_seq_len]
                    else:
                        tokens = tokens + [0] * (max_seq_len - len(tokens))
                    self.examples.append(tokens)
        
        print(f"Loaded {len(self.examples)} conversations")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        tokens = self.examples[idx]
        x = torch.tensor(tokens[:-1], dtype=torch.long)
        y = torch.tensor(tokens[1:], dtype=torch.long)
        return x, y

def get_cosine_schedule(optimizer, total_steps):
    def lr_lambda(step):
        if step < 100:
            return step / 100
        progress = (step - 100) / max(1, total_steps - 100)
        return max(0.1, 0.5 * (1 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def knoc8_finetune():
    device = torch.device('cuda')
    
    print("=" * 50)
    print("knoc8 FINE-TUNING V3")
    print("=" * 50)
    
    model = CodeSeek(VOCAB_SIZE, D_MODEL, N_HEADS, N_LAYERS, FFN_DIM, MAX_SEQ_LEN, DROPOUT)
    
    checkpoint_path = 'checkpoints/knoc8_2.pt'
    if not os.path.exists(checkpoint_path):
        checkpoint_path = 'checkpoints/best_model.pt'
    
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model.load_state_dict(ckpt)
    
    model = model.to(device)
    print(f"Loaded base model: {model.count_params()/1e6:.1f}M params")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    dataset = ChatDataset('data/fine.txt', TOKENIZER_PATH, MAX_SEQ_LEN)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, pin_memory=True)
    
    FINETUNE_LR = 5e-5
    FINETUNE_EPOCHS = 5
    GRADIENT_ACCUM = 2
    SAVE_EVERY = 500
    LOG_EVERY = 20
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=FINETUNE_LR,
        betas=(0.9, 0.95),
        weight_decay=0.1
    )
    
    total_steps = len(dataloader) * FINETUNE_EPOCHS // GRADIENT_ACCUM
    scheduler = get_cosine_schedule(optimizer, total_steps)
    scaler = torch.amp.GradScaler('cuda')
    
    os.makedirs('checkpoints/knoc8_chat_v3', exist_ok=True)
    
    print(f"\nConfig:")
    print(f"  LR: {FINETUNE_LR}")
    print(f"  Epochs: {FINETUNE_EPOCHS}")
    print(f"  Grad Accum: {GRADIENT_ACCUM}")
    print(f"  Conversations: {len(dataset)}")
    print(f"  Total Steps: {total_steps}")
    print(f"  Data Format: User:/Assistant:")
    print(f"\nTraining...\n")
    
    model.train()
    step = 0
    best_loss = float('inf')
    
    for epoch in range(FINETUNE_EPOCHS):
        total_loss = 0
        saved_this_step = False
        
        for i, (x, y) in enumerate(dataloader):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            
            with torch.amp.autocast('cuda'):
                _, loss = model(x, y)
                loss = loss / GRADIENT_ACCUM
            
            scaler.scale(loss).backward()
            
            if (i + 1) % GRADIENT_ACCUM == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                scheduler.step()
                saved_this_step = False
            
            total_loss += loss.item() * GRADIENT_ACCUM
            
            if i % LOG_EVERY == 0:
                lr = scheduler.get_last_lr()[0]
                current_loss = loss.item() * GRADIENT_ACCUM
                vram = torch.cuda.memory_allocated() / 1024**3
                print(f"E{epoch+1} S{step} L{current_loss:.4f} LR{lr:.8f} V{vram:.1f}G")
            
            if step > 0 and step % SAVE_EVERY == 0 and not saved_this_step:
                torch.save({
                    'step': step,
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, f'checkpoints/knoc8_chat_v3/step_{step}.pt')
                print(f"💾 Saved step_{step}")
                saved_this_step = True
            
            if loss.item() < best_loss:
                best_loss = loss.item()
                torch.save(model.state_dict(), 'checkpoints/knoc8_chat_v3/best_model.pt')
        
        avg_loss = total_loss / len(dataloader)
        print(f"\n✅ Epoch {epoch+1} Complete! Avg Loss: {avg_loss:.4f}\n")
        torch.save(model.state_dict(), f'checkpoints/knoc8_chat_v3/epoch_{epoch+1}.pt')
    
    torch.save(model.state_dict(), 'checkpoints/knoc8-Chat-v3.pt')
    
    print("=" * 50)
    print("🎉 FINE-TUNING COMPLETE!")
    print("Model: checkpoints/knoc8-Chat-v3.pt")
    print("=" * 50)

if __name__ == "__main__":
    knoc8_finetune()