import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tokenizers import Tokenizer
import os
import sys
import glob
import math

sys.path.insert(0, '.')
from model.transformer import CodeSeek
from config import *

class FullDataset(Dataset):
    def __init__(self, data_dir, tokenizer_path, max_seq_len):
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.max_seq_len = max_seq_len
        self.chunks = []
        
        files = glob.glob(f'{data_dir}/*.txt')
        print(f"Files: {len(files)}")
        
        for f in files:
            with open(f, 'r', encoding='utf-8') as file:
                text = file.read()
            tokens = self.tokenizer.encode(text).ids
            
            for i in range(0, len(tokens) - max_seq_len, max_seq_len):
                self.chunks.append(tokens[i:i + max_seq_len + 1])
        
        print(f"Chunks: {len(self.chunks):,}")
    
    def __len__(self):
        return len(self.chunks)
    
    def __getitem__(self, idx):
        chunk = self.chunks[idx]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y

def get_cosine_schedule(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.1, 0.5 * (1 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def train():
    device = torch.device('cuda')
    
    model = CodeSeek(VOCAB_SIZE, D_MODEL, N_HEADS, N_LAYERS, FFN_DIM, MAX_SEQ_LEN, DROPOUT)
    model = model.to(device)
    print(f"Model: {model.count_params()/1e6:.1f}M params")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    dataset = FullDataset(DATA_DIR + '/text_data', TOKENIZER_PATH, MAX_SEQ_LEN)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, pin_memory=True)
    
    total_steps = len(dataloader) * EPOCHS // GRAD_ACCUM
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )
    
    scheduler = get_cosine_schedule(optimizer, WARMUP_STEPS, total_steps)
    scaler = torch.amp.GradScaler('cuda')
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    model.train()
    step = 0
    best_loss = float('inf')
    
    for epoch in range(EPOCHS):
        total_loss = 0
        
        for i, (x, y) in enumerate(dataloader):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            
            with torch.amp.autocast('cuda'):
                logits, loss = model(x, y)
                loss = loss / GRAD_ACCUM
            
            scaler.scale(loss).backward()
            
            if (i + 1) % GRAD_ACCUM == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                step += 1
            
            total_loss += loss.item() * GRAD_ACCUM
            
            if i % LOG_EVERY == 0:
                lr = scheduler.get_last_lr()[0]
                vram = torch.cuda.memory_allocated()/1024**3
                print(f"E{epoch+1} S{step} L{loss.item()*GRAD_ACCUM:.3f} LR{lr:.6f} V{vram:.1f}G")
            
            if step > 0 and step % SAVE_EVERY == 0:
                torch.save({
                    'step': step, 'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, f'{CHECKPOINT_DIR}/step_{step}.pt')
                print(f"Saved step {step}")
            
            if loss.item() < best_loss:
                best_loss = loss.item()
                torch.save(model.state_dict(), f'{CHECKPOINT_DIR}/best_model.pt')
        
        avg = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} done. Avg Loss: {avg:.4f}\n")
        torch.save(model.state_dict(), f'{CHECKPOINT_DIR}/epoch_{epoch+1}.pt')
    
    torch.save(model.state_dict(), f'{CHECKPOINT_DIR}/codeseek_final.pt')
    print("Training Complete!")

if __name__ == "__main__":
    train()