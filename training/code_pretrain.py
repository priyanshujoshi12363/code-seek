import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import os
import sys
import math

sys.path.insert(0, '.')
from model.transformer import CodeSeek
from config import *

class CodeDataset(Dataset):
    def __init__(self, tokenized_path):
        self.chunks = torch.load(tokenized_path)
        print(f"Loaded {len(self.chunks):,} chunks")
    def __len__(self):
        return len(self.chunks)
    def __getitem__(self, idx):
        chunk = self.chunks[idx]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y

def get_cosine_schedule(optimizer, warmup_steps, total_steps, start_step=0):
    def lr_lambda(step):
        actual = step + start_step
        if actual < warmup_steps:
            return actual / max(1, warmup_steps)
        progress = (actual - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.1, 0.5 * (1 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def code_pretrain():
    device = torch.device('cuda')
    
    CODE_LR = 1e-4
    CODE_EPOCHS = 2
    CODE_WARMUP = 500
    SAVE_EVERY = 1000
    LOG_EVERY = 50
    
    model = CodeSeek(VOCAB_SIZE, D_MODEL, N_HEADS, N_LAYERS, FFN_DIM, MAX_SEQ_LEN, DROPOUT)
    
    resume_path = 'checkpoints/code/code_step_4000.pt'
    
    if os.path.exists(resume_path):
        print(f"Resuming from: {resume_path}")
        ckpt = torch.load(resume_path, map_location='cpu')
        model.load_state_dict(ckpt['model_state_dict'])
        model = model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=CODE_LR, betas=(0.9, 0.95), weight_decay=0.1)
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_step = ckpt['step']
        start_epoch = ckpt['epoch']
        print(f"Step: {start_step}, Epoch: {start_epoch+1}")
    else:
        print("Loading story model...")
        ckpt = torch.load('checkpoints/code-seek_stories_v1.pt', map_location='cpu')
        if 'model_state_dict' in ckpt:
            model.load_state_dict(ckpt['model_state_dict'])
        else:
            model.load_state_dict(ckpt)
        model = model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=CODE_LR, betas=(0.9, 0.95), weight_decay=0.1)
        start_step = 0
        start_epoch = 0
    
    print(f"Model: {model.count_params()/1e6:.1f}M params")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    dataset = CodeDataset("data/code_data/tokenized_code.pt")
    
    os.makedirs('checkpoints/code', exist_ok=True)
    scaler = torch.amp.GradScaler('cuda')
    
    model.train()
    step = start_step
    best_loss = float('inf')
    
    for epoch in range(start_epoch, CODE_EPOCHS):
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=0)
        total_steps = len(dataloader) * CODE_EPOCHS // GRAD_ACCUM
        scheduler = get_cosine_schedule(optimizer, CODE_WARMUP, total_steps, step)
        
        total_loss = 0
        saved_this_step = False
        
        for i, (x, y) in enumerate(dataloader):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            
            with torch.amp.autocast('cuda'):
                _, loss = model(x, y)
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
                saved_this_step = False
            
            total_loss += loss.item() * GRAD_ACCUM
            
            if i % LOG_EVERY == 0:
                lr = scheduler.get_last_lr()[0]
                vram = torch.cuda.memory_allocated() / 1024**3
                print(f"E{epoch+1} S{step} L{loss.item()*GRAD_ACCUM:.3f} LR{lr:.6f} V{vram:.1f}G")
            
            if step > 0 and step % SAVE_EVERY == 0 and not saved_this_step:
                torch.save({
                    'step': step,
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, f'checkpoints/code/code_step_{step}.pt')
                print(f"Saved code_step_{step}")
                saved_this_step = True
            
            if loss.item() < best_loss:
                best_loss = loss.item()
                torch.save(model.state_dict(), 'checkpoints/code/best_code_model.pt')
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} done. Avg Loss: {avg_loss:.4f}")
        torch.save(model.state_dict(), f'checkpoints/code/code_epoch_{epoch+1}.pt')
    
    torch.save(model.state_dict(), 'checkpoints/CodeSeek-Code-v1.pt')
    print("Code pretraining complete!")

if __name__ == "__main__":
    code_pretrain()