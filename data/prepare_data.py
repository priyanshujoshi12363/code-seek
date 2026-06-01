import os

data_dir = "data/text_data"

for f in os.listdir(data_dir):
    if f.endswith('.txt'):
        path = os.path.join(data_dir, f)
        size = os.path.getsize(path) / 1024 / 1024
        
        if size > 50:
            print(f"Splitting {f} ({size:.1f}MB)...")
            
            with open(path, 'r', encoding='utf-8') as file:
                text = file.read()
            
            os.remove(path)
            
            chunk_size = 500000
            for i, start in enumerate(range(0, len(text), chunk_size)):
                chunk = text[start:start + chunk_size]
                new_name = f.replace('.txt', f'_{i}.txt')
                with open(os.path.join(data_dir, new_name), 'w', encoding='utf-8') as out:
                    out.write(chunk)
            
            print(f"  → {i+1} files created")

print("\nDone! Check files:")
for f in os.listdir(data_dir):
    if f.endswith('.txt'):
        size = os.path.getsize(os.path.join(data_dir, f)) / 1024
        print(f"  {f}: {size:.1f} KB")