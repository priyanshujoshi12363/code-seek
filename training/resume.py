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
        return torch.tensor(chunk[:-1], dtype=torch.long), torch.tensor(chunk[1:], dtype=torch.long)

def get_cosine_schedule(optimizer, warmup_steps, total_steps, start_step):
    def lr_lambda(step):
        actual = step + start_step
        if actual < warmup_steps:
            return actual / max(1, warmup_steps)
        progress = (actual - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.1, 0.5 * (1 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def resume():
    device = torch.device('cuda')
    
    model = CodeSeek(VOCAB_SIZE, D_MODEL, N_HEADS, N_LAYERS, FFN_DIM, MAX_SEQ_LEN, DROPOUT)
    model = model.to(device)
    print(f"Model: {model.count_params()/1e6:.1f}M params")
    
    dataset = FullDataset(DATA_DIR + '/text_data', TOKENIZER_PATH, MAX_SEQ_LEN)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, pin_memory=True)
    
    total_steps = len(dataloader) * EPOCHS // GRAD_ACCUM
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.95), weight_decay=0.1)
    scaler = torch.amp.GradScaler('cuda')
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    ckpt = torch.load(f'{CHECKPOINT_DIR}/knoc8.pt', map_location='cpu')
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    start_step = ckpt['step']
    start_epoch = ckpt['epoch']
    print(f"Resumed from Step {start_step}")
    
    scheduler = get_cosine_schedule(optimizer, WARMUP_STEPS, total_steps, start_step)
    
    model.train()
    step = start_step
    saved = False
    
    for epoch in range(start_epoch, EPOCHS):
        total_loss = 0
        for i, (x, y) in enumerate(dataloader):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
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
                saved = False
            total_loss += loss.item() * GRAD_ACCUM
            if i % LOG_EVERY == 0:
                lr = scheduler.get_last_lr()[0]
                print(f"E{epoch+1} S{step} L{loss.item()*GRAD_ACCUM:.3f} LR{lr:.6f}")
            if step > 0 and step % SAVE_EVERY == 0 and not saved:
                torch.save({'step':step,'epoch':epoch,'model_state_dict':model.state_dict(),'optimizer_state_dict':optimizer.state_dict()}, f'{CHECKPOINT_DIR}/step_{step}.pt')
                print(f"Saved {step}")
                saved = True
        print(f"Epoch {epoch+1} done. Avg: {total_loss/len(dataloader):.4f}")
        torch.save(model.state_dict(), f'{CHECKPOINT_DIR}/epoch_{epoch+1}.pt')
    
    torch.save(model.state_dict(), f'{CHECKPOINT_DIR}/final.pt')
    print("Done!")

if __name__ == "__main__":
    resume()