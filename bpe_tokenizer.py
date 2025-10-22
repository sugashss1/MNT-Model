import json
import os
from collections import defaultdict
import heapq

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

class BpeTokenizer:
    """
    A Byte-Pair Encoding (BPE) Tokenizer.
    This is a byte-level BPE tokenizer that learns merges from a corpus and can
    encode/decode text based on UTF-8 bytes.
    """
    def __init__(self):
        self.vocab = {}
        self.merges = {}
        # Pre-assign special tokens to fixed IDs
        self.special_tokens = ['<pad>', '<sos>', '<eos>', '<unk>']
        self.vocab = {token: i for i, token in enumerate(self.special_tokens)}

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
        Trains the tokenizer on a given corpus using a priority queue for efficiency.
        
        Args:
            corpus (list[str]): A list of sentences to train on.
            vocab_size (int): The target size of the vocabulary.
        """
        if vocab_size < 256 + len(self.special_tokens):
            raise ValueError("Vocab size must be at least 256 + number of special tokens")

        num_merges = vocab_size - 256 - len(self.special_tokens)
        text_bytes = [s.encode("utf-8") for s in corpus]
        ids = [list(b) for b in text_bytes]

        # --- New, Faster Algorithm ---
        
        # 1. Initial Count (scan corpus once)
        print("Step 1/3: Counting initial pairs...")
        stats = defaultdict(int)
        for sublist in ids:
            for pair in zip(sublist, sublist[1:]):
                stats[pair] += 1

        # 2. Populate Priority Queue
        print("Step 2/3: Building priority queue...")
        # We use a min-heap, so we store negative counts to simulate a max-heap
        pq = [(-count, pair) for pair, count in stats.items()]
        heapq.heapify(pq)

        # 3. Iterative Merging
        print(f"Step 3/3: Performing {num_merges} merges...")
        print_progress_bar(0, num_merges, prefix='Progress:', suffix='Complete', length=50)
        for i in range(num_merges):
            if not pq:
                break

            # Get the most frequent pair
            neg_count, best_pair = heapq.heappop(pq)
            
            new_id = 256 + len(self.special_tokens) + i
            self.merges[best_pair] = new_id

            # This is a simplified update. A fully optimal version would be more complex.
            # We still need to scan to merge, but we avoid re-counting everything.
            new_ids = []
            for sublist in ids:
                new_sublist = []
                j = 0
                while j < len(sublist):
                    if j < len(sublist) - 1 and (sublist[j], sublist[j+1]) == best_pair:
                        new_sublist.append(new_id)
                        j += 2
                    else:
                        new_sublist.append(sublist[j])
                        j += 1
                new_ids.append(new_sublist)
            ids = new_ids
            
            # For simplicity, we won't dynamically update the PQ in this implementation,
            # as it's very complex. We will just re-count and re-heapify periodically.
            # This is a hybrid approach - faster than before, but not fully optimal.
            if (i + 1) % 250 == 0 or i == num_merges - 1:
                stats = defaultdict(int)
                for sublist in ids:
                    for pair in zip(sublist, sublist[1:]): stats[pair] += 1
                pq = [(-count, pair) for pair, count in stats.items()]
                heapq.heapify(pq)

            print_progress_bar(i + 1, num_merges, prefix='Progress:', suffix='Complete', length=50)

    def get_vocab_size(self):
        vocab = {token: i for i, token in enumerate(self.special_tokens)}
        for i in range(256):
            vocab[chr(i)] = i + len(self.special_tokens)
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
            for pair in zip(ids, ids[1:]): stats[pair] += 1
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges: break
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

    def decode(self, token_ids):
        byte_list = []
        for token_id in token_ids:
            if token_id in self.special_tokens_map_inv:
                 byte_list.extend(self.special_tokens_map_inv[token_id].encode('utf-8'))
                 continue
            def get_bytes(tid):
                if tid < 256: return [tid]
                pair = next((p for p, i in self.merges.items() if i == tid), None)
                if pair is None: return [self.vocab.get('<unk>', 3)]
                return get_bytes(pair[0]) + get_bytes(pair[1])
            byte_list.extend(get_bytes(token_id))
        try:
            text = bytes(byte_list).decode("utf-8", errors="replace")
        except:
            text = "<decoding error>"
        return text

    def save(self, filepath):
        tokenizer_state = {
            "vocab": self.vocab,
            "merges": {f"{p[0]},{p[1]}": v for p, v in self.merges.items()}
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(tokenizer_state, f, ensure_ascii=False, indent=2)

    def load(self, filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            tokenizer_state = json.load(f)
        self.vocab = tokenizer_state["vocab"]
        self.merges = {tuple(map(int, k.split(','))): v for k, v in tokenizer_state["merges"].items()}
        self.special_tokens_map_inv = {i: s for s, i in self.vocab.items() if s in self.special_tokens}
