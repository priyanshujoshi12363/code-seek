import os
import glob
from tokenizers import Tokenizer, models, trainers, pre_tokenizers

def train_tokenizer():
    data_dir = "data/text_data"
    files = glob.glob(f'{data_dir}/*.txt')
    
    if not files:
        print("No text files found. Run download_data.py first")
        return
    
    print(f"Found {len(files)} files")
    
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    
    trainer = trainers.BpeTrainer(
        vocab_size=16384,
        special_tokens=["<PAD>", "<UNK>", "<EOS>", "<BOS>"],
        min_frequency=2,
    )
    
    tokenizer.train(files, trainer)
    os.makedirs("tokenizer", exist_ok=True)
    tokenizer.save("tokenizer/codeseek_tokenizer.json")
    
    print(f"Done. Vocab: {tokenizer.get_vocab_size()}")
    
    for text in ["Hello world", "int main() { return 0; }"]:
        tokens = tokenizer.encode(text).tokens
        print(f"  '{text}' -> {tokens}")

if __name__ == "__main__":
    train_tokenizer()