import os
import shutil
import argparse
import numpy as host_np # Use host numpy for file I/O
import random

from bpe_tokenizer import BpeTokenizer

def print_progress_bar(iteration, total, prefix='', suffix='', length=50, fill='█'):
    """
    Call in a loop to create a terminal progress bar.
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:.1f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='\r')
    if iteration == total: 
        print()

def load_corpus(en_path, ta_path):
    """Loads parallel corpus from two files."""
    with open(en_path, 'r', encoding='utf-8') as f_en, \
         open(ta_path, 'r', encoding='utf-8') as f_ta:
        en_lines = [line.strip() for line in f_en.readlines()]
        ta_lines = [line.strip() for line in f_ta.readlines()]
    
    assert len(en_lines) == len(ta_lines), "Mismatched number of lines in corpus files!"
    corpus = list(zip(en_lines, ta_lines))
    print(f"Loaded {len(corpus)} sentence pairs.")
    return corpus

def train_tokenizer_if_needed(corpus_files, tokenizer_path, vocab_size, force):
    """Trains the BPE tokenizer if it doesn't exist."""
    if os.path.exists(tokenizer_path) and not force:
        print(f"Tokenizer already exists at '{tokenizer_path}'. Skipping training.")
        return
    
    print("--- Starting Tokenizer Training ---")
    
    # Load all sentences from all corpus files for training
    corpus = []
    for path in corpus_files:
        with open(path, 'r', encoding='utf-8') as f:
            corpus.extend([line.strip() for line in f.readlines() if line.strip()])
    
    # --- COMPLEXITY IMPROVEMENT: Sub-sample the corpus for faster training ---
    # Training on the full dataset is slow and often unnecessary.
    # A large enough random sample is sufficient.
    sample_size = min(100000, len(corpus)) # Use 100k sentences or the whole corpus if smaller
    print(f"Loaded {len(corpus)} total sentences. Using a random sample of {sample_size} for training.")
    corpus = random.sample(corpus, sample_size)

    tokenizer = BpeTokenizer()
    tokenizer.train(corpus, vocab_size=vocab_size)
    tokenizer.save(tokenizer_path)
    print(f"\nTokenizer with vocab size ~{vocab_size} trained and saved to '{tokenizer_path}'")

def create_and_save_batches(corpus, tokenizer, batch_size, max_seq_len, save_dir, force_reprocess):
    """
    Pre-processes the entire corpus, creates padded batches, and saves them to disk.
    """
    if force_reprocess and os.path.exists(save_dir):
        print(f"Force re-processing enabled. Deleting existing batches in '{save_dir}'...")
        shutil.rmtree(save_dir)
    elif os.path.exists(save_dir) and os.listdir(save_dir):
        print(f"Found pre-processed batches in '{save_dir}'. Use --force to re-process.")
        return

    print(f"Starting pre-processing...")
    os.makedirs(save_dir, exist_ok=True)

    processed_data = []
    total_sentences = len(corpus)
    print(f"Tokenizing {total_sentences} sentences...")
    print_progress_bar(0, total_sentences, prefix='Progress:', suffix='Complete', length=50)
    for i, (src_text, tgt_text) in enumerate(corpus):
        src_seq = tokenizer.encode(src_text)[:max_seq_len]
        tgt_seq = tokenizer.encode(tgt_text)[:max_seq_len - 1]
        tgt_input = [tokenizer.vocab.get('<sos>', 1)] + tgt_seq
        tgt_expected = tgt_seq + [tokenizer.vocab.get('<eos>', 2)]
        processed_data.append((src_seq, tgt_input, tgt_expected))
        # Update progress bar
        print_progress_bar(i + 1, total_sentences, prefix='Progress:', suffix='Complete', length=50)

    # Sort by source sentence length to create batches with similar lengths, which reduces padding.
    processed_data.sort(key=lambda x: len(x[0]))
    print("Tokenized and sorted all data by length.")

    pad_id = tokenizer.vocab.get('<pad>', 0)
    print("Creating and saving batches...")
    num_batches = (len(processed_data) + batch_size - 1) // batch_size
    print_progress_bar(0, num_batches, prefix='Progress:', suffix='Complete', length=50)
    for i in range(num_batches):
        batch = processed_data[i*batch_size:(i+1)*batch_size]
        if not batch: continue
        
        src_seqs, tgt_inputs, tgt_expected = zip(*batch)
        
        max_src_len = max(len(s) for s in src_seqs)
        max_tgt_len = max(len(s) for s in tgt_inputs)
        
        src_padded = host_np.array([s + [pad_id] * (max_src_len - len(s)) for s in src_seqs], dtype=host_np.int32)
        tgt_padded = host_np.array([s + [pad_id] * (max_tgt_len - len(s)) for s in tgt_inputs], dtype=host_np.int32)
        expected_padded = host_np.array([s + [pad_id] * (max_tgt_len - len(s)) for s in tgt_expected], dtype=host_np.int32)
        
        batch_filename = os.path.join(save_dir, f"batch_{i:05d}.npz")
        host_np.savez_compressed(batch_filename, src=src_padded, tgt=tgt_padded, expected=expected_padded)
        
        print_progress_bar(i + 1, num_batches, prefix='Progress:', suffix='Complete', length=50)

    print(f"Saved {num_batches} pre-processed batches to '{save_dir}'.")

def main():
    parser = argparse.ArgumentParser(description="Pre-process data for the Transformer model.")
    parser.add_argument('--force', action='store_true', help="Force re-processing of the dataset, overwriting existing batches.")
    args = parser.parse_args()

    # --- Configuration --- # I've reduced vocab size for faster testing
    VOCAB_SIZE = 4000
    BATCH_SIZE = 4
    # MAX_SEQ_LENGTH should match the one in train.py
    MAX_SEQ_LENGTH = 128
    CORPUS_EN_PATH = 'data1-5.en'
    CORPUS_TA_PATH = 'data1-5.ta'
    TOKENIZER_PATH = "bpe_tokenizer.json"
    BATCH_SAVE_DIR = "processed_batches"

    # 1. Train tokenizer if it doesn't exist
    train_tokenizer_if_needed([CORPUS_EN_PATH, CORPUS_TA_PATH], TOKENIZER_PATH, VOCAB_SIZE, args.force)

    # 2. Load tokenizer and corpus
    tokenizer = BpeTokenizer()
    tokenizer.load(TOKENIZER_PATH)
    train_corpus = load_corpus(CORPUS_EN_PATH, CORPUS_TA_PATH)

    # 3. Create and save batches
    create_and_save_batches(train_corpus, tokenizer, BATCH_SIZE, MAX_SEQ_LENGTH, BATCH_SAVE_DIR, args.force)
    
    print("\n--- Data pre-processing complete! ---")

if __name__ == "__main__":
    main()