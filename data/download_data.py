from datasets import load_dataset
print("Downloading alternative wiki...")
ds = load_dataset("olm/wikipedia", "2022-12", split="train", streaming=True)

with open("data/text_data/wiki.txt", "w", encoding="utf-8") as f:
    count = 0
    for item in ds:
        text = item.get('text', '').strip()
        if len(text) > 200:
            f.write(text + "\n\n")
            count += 1
        if count >= 10000:
            break

print(f"Done: {count} articles")