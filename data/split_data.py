"""Split all story text in data/text_data/ into many files of 5000 lines each.

Reads every existing stories_*.txt shard, concatenates the lines, then writes
them back out as part_000.txt, part_001.txt, ... each containing 5000 lines.
The original shards are removed afterwards so training only sees the parts.
"""
import os
import glob

DATA_DIR = "data/text_data"
LINES_PER_FILE = 5000

# gather all lines from the current shards
shards = sorted(glob.glob(os.path.join(DATA_DIR, "stories_*.txt")))
print(f"Reading {len(shards)} shard(s)...")

lines = []
for path in shards:
    with open(path, "r", encoding="utf-8") as f:
        lines.extend(f.readlines())

print(f"Total lines: {len(lines):,}")

# write out in 5000-line parts
num_parts = 0
for i in range(0, len(lines), LINES_PER_FILE):
    chunk = lines[i:i + LINES_PER_FILE]
    out_path = os.path.join(DATA_DIR, f"part_{num_parts:03d}.txt")
    with open(out_path, "w", encoding="utf-8") as out:
        out.writelines(chunk)
    num_parts += 1

# remove the original shards so the dataset is only the 5000-line parts
for path in shards:
    os.remove(path)

print(f"Done: wrote {num_parts} files of up to {LINES_PER_FILE} lines each into {DATA_DIR}")
