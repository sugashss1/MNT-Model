import time
import json
from collections import defaultdict
import random

# --- GPU/Device Setup ---
try:
    import cupy as np
    np.cuda.runtime.getDeviceCount() # Check for GPU
    print("--- Using CuPy for GPU acceleration ---")
    IS_GPU = True
except (ImportError, np.cuda.runtime.CUDARuntimeError):
    import numpy as np
    print("--- CuPy not found or no GPU available, using NumPy for CPU ---")
    IS_GPU = False

# Import the custom modules we've built
from autograd import Tensor
from numpy_layers import Transformer
from training_utils import CrossEntropyLoss, Adam

# --- BPE Tokenizer Class ---
# Including the class directly in this script to ensure the correct version is used.
class BpeTokenizer:
    """
    A BPE Tokenizer that learns merges from a corpus and can encode/decode text.
    """
    def __init__(self):
        self.vocab = {}
        self.merges = {}
        # Pre-assign special tokens to fixed IDs
        self.special_tokens = ['<pad>', '<sos>', '<eos>', '<unk>']
        self.vocab = {token: i for i, token in enumerate(self.special_tokens)}

    def _get_stats(self, ids):
        """Counts the frequency of all adjacent pairs of integers in a list of lists."""
        counts = defaultdict(int)
        for sublist in ids:
            for pair in zip(sublist, sublist[1:]):
                counts[pair] += 1
        return counts

    def _merge(self, ids, pair, idx):
        """Replaces all occurrences of a pair in a list of lists with a new integer."""
        new_ids = []
        for sublist in ids:
            new_sublist = []
            i = 0
            while i < len(sublist):
                if i < len(sublist) - 1 and (sublist[i], sublist[i+1]) == pair:
                    new_sublist.append(idx)
                    i += 2
                else:
                    new_sublist.append(sublist[i])
                    i += 1
            new_ids.append(new_sublist)
        return new_ids

    def train(self, corpus: list, vocab_size: int):
        """
        Trains the tokenizer on a given corpus.
        
        Args:
            corpus (list[str]): A list of sentences to train on.
            vocab_size (int): The target size of the vocabulary.
        """
        if vocab_size < 256 + len(self.special_tokens):
            raise ValueError("Vocab size must be at least 256 + number of special tokens")

        # 1. Initial vocabulary: all bytes (0-255) and special tokens
        num_merges = vocab_size - 256 - len(self.special_tokens)
        
        # 2. Pre-tokenize the corpus into sequences of UTF-8 byte integers
        text_bytes = [s.encode("utf-8") for s in corpus]
        ids = [list(b) for b in text_bytes]

        # 3. Iteratively merge the most frequent pair of tokens
        for i in range(num_merges):
            stats = self._get_stats(ids)
            if not stats:
                break
            
            best_pair = max(stats, key=stats.get)
            new_id = 256 + len(self.special_tokens) + i
            ids = self._merge(ids, best_pair, new_id)
            self.merges[best_pair] = new_id

    def get_vocab_size(self):
        # Build vocab dynamically from merges to ensure correctness
        vocab = {token: i for i, token in enumerate(self.special_tokens)}
        # Add bytes 0-255
        for i in range(256):
            vocab[chr(i)] = i + len(self.special_tokens)

        # Add merged tokens
        temp_merges_vocab = {}
        for (p0, p1), idx in self.merges.items():
            s0 = temp_merges_vocab.get(p0, chr(p0))
            s1 = temp_merges_vocab.get(p1, chr(p1))
            temp_merges_vocab[idx] = s0 + s1
            vocab[s0 + s1] = idx + len(self.special_tokens)
        
        self.vocab = {k:v for k,v in sorted(vocab.items(), key=lambda item: item[1])}
        return len(self.vocab)


    def encode(self, text: str) -> list:
        ids = list(text.encode("utf-8"))
        while len(ids) >= 2:
            stats = defaultdict(int)
            for pair in zip(ids, ids[1:]):
                stats[pair] += 1
            
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            
            idx = self.merges[pair]
            new_ids = []
            i = 0
            while i < len(ids):
                if i < len(ids) - 1 and (ids[i], ids[i+1]) == pair:
                    new_ids.append(idx)
                    i += 2
                else:
                    new_ids.append(ids[i])
                    i += 1
            ids = new_ids
        return ids

    def decode(self, ids: list) -> str:
        byte_list = []
        for token_id in ids:
            if token_id in self.special_tokens_map_inv:
                 byte_list.extend(self.special_tokens_map_inv[token_id].encode('utf-8'))
                 continue

            def get_bytes(tid):
                if tid < 256:
                    return [tid]
                pair = None
                for p, i in self.merges.items():
                    if i == tid:
                        pair = p
                        break
                if pair is None: return [self.vocab.get('<unk>', 3)]
                return get_bytes(pair[0]) + get_bytes(pair[1])
            byte_list.extend(get_bytes(token_id))
        
        try:
            text = bytes(byte_list).decode("utf-8", errors="replace")
        except:
            text = "<decoding error>"
        return text

    def save(self, file_path):
        model = {
            "vocab": self.vocab,
            "merges": {f"{p[0]},{p[1]}": v for p, v in self.merges.items()}
        }
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(model, f, ensure_ascii=False, indent=2)

    def load(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            model = json.load(f)
        self.vocab = model["vocab"]
        self.merges = {tuple(map(int, k.split(','))): v for k, v in model["merges"].items()}
        self.special_tokens_map_inv = {i: s for s, i in self.vocab.items() if s in self.special_tokens}


def get_params(model):
    """ Helper function to collect all model parameters """
    params = []
    if isinstance(model, Transformer):
        modules = [
            model.encoder_embedding, model.decoder_embedding,
            *model.encoder_layers, *model.decoder_layers,
            model.fc
        ]
        for module in modules:
            params.extend(get_params(module))
    elif hasattr(model, 'params'):
        params.extend(model.params)
    return params

def zero_all_grads(model):
    """ Helper function to zero out all gradients in the model """
    for p in get_params(model):
        p.zero_grad()

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

if __name__ == "__main__":
    print("--- Starting Transformer Training ---")

    # 1. Hyperparameters
    D_MODEL = 64
    NUM_HEADS = 4
    NUM_LAYERS = 3
    D_FF = 128
    MAX_SEQ_LENGTH = 256 # Reduced from a potentially larger value
    DROPOUT = 0.1
    LEARNING_RATE = 0.0005
    EPOCHS = 10
    BATCH_SIZE = 8 # Reduced from 16
    VOCAB_SIZE = 5000

    # 2. Load Data
    print("\nLoading training data...")
    train_corpus = load_corpus('data1-5.en', 'data1-5.ta')
    
    # 3. Train or Load Tokenizer
    tokenizer_path = "bpe_tokenizer.json"
    tokenizer = BpeTokenizer()
    try:
        tokenizer.load(tokenizer_path)
        print(f"Loaded tokenizer from {tokenizer_path}")
    except FileNotFoundError:
        print("\nTraining Tokenizer...")
        english_sentences = [pair[0] for pair in train_corpus]
        tamil_sentences = [pair[1] for pair in train_corpus]
        tokenizer.train(english_sentences + tamil_sentences, vocab_size=VOCAB_SIZE)
        tokenizer.save(tokenizer_path)
        print(f"Tokenizer trained and saved to {tokenizer_path}")

    SRC_VOCAB_SIZE = TGT_VOCAB_SIZE = tokenizer.get_vocab_size()
    print(f"Vocabulary size: {SRC_VOCAB_SIZE}")

    # 4. Initialize Model, Loss, and Optimizer
    print("\nInitializing Model...")
    model = Transformer(
        src_vocab_size=SRC_VOCAB_SIZE,
        tgt_vocab_size=TGT_VOCAB_SIZE,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        d_ff=D_FF,
        max_seq_length=MAX_SEQ_LENGTH,
        dropout_p=DROPOUT
    )
    loss_fn = CrossEntropyLoss()
    optimizer = Adam(get_params(model), lr=LEARNING_RATE)
    print("Model, Loss, and Optimizer initialized.")

    # 5. The Training Loop
    print("\nStarting Training Loop...")
    start_time = time.time()
    for epoch in range(EPOCHS):
        epoch_loss = 0
        random.shuffle(train_corpus)
        
        for i in range(0, len(train_corpus), BATCH_SIZE):
            batch = train_corpus[i:i+BATCH_SIZE]
            
            src_texts = [pair[0] for pair in batch]
            tgt_texts = [pair[1] for pair in batch]

            # --- Prepare Batch Data ---
            src_seqs = [tokenizer.encode(s) for s in src_texts]
            tgt_seqs = [tokenizer.encode(s) for s in tgt_texts]

            # CORRECTED: Truncate sequences *before* padding
            src_seqs = [s[:MAX_SEQ_LENGTH] for s in src_seqs]
            tgt_seqs = [s[:MAX_SEQ_LENGTH-1] for s in tgt_seqs] # Leave room for sos/eos

            tgt_inputs = [[tokenizer.vocab['<sos>']] + s for s in tgt_seqs]
            tgt_expected = [s + [tokenizer.vocab['<eos>']] for s in tgt_seqs]

            max_src_len = max(len(s) for s in src_seqs)
            max_tgt_len = max(len(s) for s in tgt_inputs)
            
            pad_id = tokenizer.vocab.get('<pad>', 0)
            # Create arrays on CPU first, as tokenizer output is standard Python lists
            import numpy as host_np 
            src_padded_cpu = host_np.array([s + [pad_id] * (max_src_len - len(s)) for s in src_seqs])
            tgt_padded_cpu = host_np.array([s + [pad_id] * (max_tgt_len - len(s)) for s in tgt_inputs])
            expected_padded_cpu = host_np.array([s + [pad_id] * (max_tgt_len - len(s)) for s in tgt_expected])

            # Move data to GPU if using CuPy
            src_padded = np.asarray(src_padded_cpu)
            tgt_padded = np.asarray(tgt_padded_cpu)
            expected_padded = np.asarray(expected_padded_cpu)

            # --- Forward Pass ---
            output = model.forward(Tensor(src_padded), Tensor(tgt_padded))

            # --- Calculate Loss ---
            output_reshaped = output.reshape(-1, SRC_VOCAB_SIZE)
            expected_reshaped = expected_padded.flatten()
            
            loss = loss_fn.forward(output_reshaped, expected_reshaped)
            epoch_loss += loss.data()

            # --- Backward Pass ---
            zero_all_grads(model)
            loss.backward()

            # --- Update Weights ---
            optimizer.step()

        avg_epoch_loss = epoch_loss / (len(train_corpus) / BATCH_SIZE)
        print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {avg_epoch_loss:.4f}")

    end_time = time.time()
    print(f"\nTraining finished in {end_time - start_time:.2f} seconds.")

    # 6. Evaluation on Test Set
    print("\n--- Evaluating on Test Set ---")
    test_corpus = load_corpus('data.en6', 'data.ta6')

    # Set all dropout layers in the model to evaluation mode
    for p in get_params(model):
        # This is a bit of a hack; a better way would be to have a proper
        # model.eval() method that traverses the module tree.
        if hasattr(p, 'eval'):
            p.eval()
    # Also set dropout on the main transformer class if it has one
    if hasattr(model, 'dropout') and hasattr(model.dropout, 'eval'):
        model.dropout.eval()


    def translate(sentence):
        src_seq = tokenizer.encode(sentence)[:MAX_SEQ_LENGTH]
        if not src_seq: return "<empty input>"
        
        # Ensure input is on the correct device (GPU/CPU)
        src_array = np.asarray([src_seq])
        src_tensor = Tensor(src_array)
        tgt_seq = [tokenizer.vocab.get('<sos>', 1)]
        
        for _ in range(MAX_SEQ_LENGTH):
            tgt_array = np.asarray([tgt_seq])
            tgt_tensor = Tensor(tgt_array)
            with np.errstate(all='ignore'):
                output = model.forward(src_tensor, tgt_tensor)
            
            next_token_id = np.argmax(output.data()[0, -1, :])
            if next_token_id == tokenizer.vocab.get('<eos>', 2) or len(tgt_seq) >= MAX_SEQ_LENGTH:
                break
            tgt_seq.append(next_token_id)
            
        return tokenizer.decode(tgt_seq[1:])

    for i in range(min(5, len(test_corpus))):
        src_sentence, tgt_sentence = test_corpus[i]
        translation = translate(src_sentence)
        print("-" * 20)
        print(f"SOURCE:    {src_sentence}")
        print(f"TARGET:    {tgt_sentence}")
        print(f"PREDICTED: {translation}")
