
import os
# datasets tries to import torch for streaming worker-sharing; this venv's torch
# binary is built for a different Python and won't import. We don't need torch to
# download data, so disable the integration entirely.
os.environ["USE_TORCH"] = "0"
os.environ["USE_TF"] = "0"
from datasets import load_dataset

OUT_DIR = "data/text_data"
TARGET_BYTES = 200 * 1024 * 1024      # ~200MB of stories
SHARD_BYTES = 45 * 1024 * 1024        # keep each file under the 50MB split limit
MIN_LEN = 100                          # skip trivially short stories

os.makedirs(OUT_DIR, exist_ok=True)

print("Streaming roneneldan/TinyStories (train split)...")
ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)

total = 0          # total bytes written
shard_idx = 0
shard_bytes = 0
count = 0
out = open(os.path.join(OUT_DIR, f"stories_{shard_idx}.txt"), "w", encoding="utf-8")

for item in ds:
    text = (item.get("text") or "").strip()
    if len(text) < MIN_LEN:
        continue

    block = text + "\n\n"
    nbytes = len(block.encode("utf-8"))

    # roll over to a new shard when the current one is full
    if shard_bytes + nbytes > SHARD_BYTES:
        out.close()
        shard_idx += 1
        shard_bytes = 0
        out = open(os.path.join(OUT_DIR, f"stories_{shard_idx}.txt"), "w", encoding="utf-8")

    out.write(block)
    shard_bytes += nbytes
    total += nbytes
    count += 1

    if count % 20000 == 0:
        print(f"  {count:,} stories | {total/1024/1024:.1f} MB")

    if total >= TARGET_BYTES:
        break

out.close()
print(f"\nDone: {count:,} stories, {total/1024/1024:.1f} MB across {shard_idx+1} file(s) in {OUT_DIR}")
