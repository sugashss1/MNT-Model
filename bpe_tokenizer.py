import json
from collections import defaultdict

class BpeTokenizer:
    """
    A Byte-Pair Encoding (BPE) Tokenizer.

    This tokenizer learns a vocabulary of sub-word units from a corpus and can
    then tokenize new text into sequences of these learned units.
    """
    def __init__(self, vocab_size=10000):
        """
        Initializes the BPE Tokenizer.

        Args:
            vocab_size (int): The target size of the vocabulary to be learned.
        """
        self.vocab_size = vocab_size
        self.vocab = {"<pad>": 0, "<sos>": 1, "<eos>": 2, "<unk>": 3} # Special tokens
        self.merges = {}
        # Inverse vocab for decoding, will be populated after training
        self.ivocab = {}

    def _get_word_stats(self, corpus):
        """
        Pre-tokenizes corpus and counts initial word frequencies.
        """
        # Splits corpus into words and counts frequency of each word.
        # Adds a special </w> token to mark the end of a word. This helps the
        # model distinguish between a sub-word in the middle of a word and
        # one at the end (e.g., 'es' vs. 'es</w>').
        word_counts = defaultdict(int)
        for sentence in corpus:
            for word in sentence.strip().split():
                word_counts[word + '</w>'] += 1
        return word_counts

    def _get_pair_stats(self, word_counts):
        """
        Calculates the frequency of all adjacent pairs of symbols.
        """
        pair_counts = defaultdict(int)
        for word, count in word_counts.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pair_counts[(symbols[i], symbols[i+1])] += count
        return pair_counts

    def _merge_pair(self, a, b, word_counts):
        """
        Merges the most frequent pair (a, b) into a new symbol 'ab'.
        """
        merged_word_counts = defaultdict(int)
        for word, count in word_counts.items():
            # Replace the pair 'a b' with the merged 'ab' in all words
            new_word = word.replace(f'{a} {b}', f'{a}{b}')
            merged_word_counts[new_word] = count
        return merged_word_counts

    def train(self, corpus):
        """
        Trains the tokenizer on a given corpus.

        Args:
            corpus (list of str): A list of sentences to train on.
        """
        print("Starting BPE training...")
        # 1. Initialize vocabulary with all individual characters
        char_vocab = set()
        for sentence in corpus:
            for char in sentence:
                char_vocab.add(char)

        for char in sorted(list(char_vocab)):
            if char not in self.vocab:
                self.vocab[char] = len(self.vocab)

        word_counts = self._get_word_stats(corpus)

        # 2. Split each word into characters for the initial state
        initial_splits = {}
        for word, count in word_counts.items():
            initial_splits[" ".join(list(word))] = count
        word_counts = initial_splits

        # 3. Iteratively merge the most frequent pair
        num_merges = self.vocab_size - len(self.vocab)
        for i in range(num_merges):
            pair_counts = self._get_pair_stats(word_counts)
            if not pair_counts:
                break # No more pairs to merge
            
            best_pair = max(pair_counts, key=pair_counts.get)
            word_counts = self._merge_pair(best_pair[0], best_pair[1], word_counts)

            # Store the merge rule and add the new token to the vocabulary
            new_token = "".join(best_pair)
            self.merges[best_pair] = new_token
            if new_token not in self.vocab:
                 self.vocab[new_token] = len(self.vocab)
            
            if (i + 1) % 100 == 0:
                print(f"  Merge {i+1}/{num_merges}: Merged {best_pair} into {new_token}. Vocab size: {len(self.vocab)}")
        
        # Populate the inverse vocabulary for decoding
        self.ivocab = {i: s for s, i in self.vocab.items()}
        print("BPE Training complete!")

    def tokenize(self, text):
        """
        Tokenizes a string of text into a sequence of token IDs.
        """
        # Add special start-of-sentence and end-of-sentence tokens
        token_ids = [self.vocab["<sos>"]]
        
        words = [word + '</w>' for word in text.strip().split()]
        
        for word in words:
            # Start with word split into characters
            symbols = " ".join(list(word))
            
            # Iteratively apply learned merges in the order they were learned
            for pair, merged in self.merges.items():
                symbols = symbols.replace(f'{pair[0]} {pair[1]}', merged)

            # Convert sub-word strings to token IDs
            for symbol in symbols.split():
                token_ids.append(self.vocab.get(symbol, self.vocab["<unk>"]))

        token_ids.append(self.vocab["<eos>"])
        return token_ids

    def decode(self, token_ids):
        """
        Decodes a sequence of token IDs back into a string.
        """
        # Convert IDs to string tokens
        tokens = [self.ivocab.get(i, "<unk>") for i in token_ids]
        
        # Filter out special tokens
        special_tokens = ["<pad>", "<sos>", "<eos>", "<unk>"]
        tokens = [t for t in tokens if t not in special_tokens]
        
        # Join tokens and handle the end-of-word markers
        text = "".join(tokens).replace('</w>', ' ')
        return text.strip()

    def save(self, filepath):
        """
        Saves the tokenizer's state (vocab and merges) to a file.
        """
        tokenizer_state = {
            "vocab": self.vocab,
            "merges": {f"{k[0]} {k[1]}": v for k, v in self.merges.items()} # Convert tuple keys to string
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(tokenizer_state, f, ensure_ascii=False, indent=2)
        print(f"Tokenizer saved to {filepath}")

    def load(self, filepath):
        """
        Loads the tokenizer's state from a file.
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            tokenizer_state = json.load(f)
        
        self.vocab = tokenizer_state["vocab"]
        # Convert string keys back to tuples
        self.merges = {tuple(k.split()): v for k, v in tokenizer_state["merges"].items()}
        self.ivocab = {i: s for s, i in self.vocab.items()}
        print(f"Tokenizer loaded from {filepath}")
