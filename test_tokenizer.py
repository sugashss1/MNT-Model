from bpe_tokenizer import BpeTokenizer
import os

def test_tokenizer():
    """
    A comprehensive test for the BpeTokenizer.
    This tests the byte-level BPE implementation.
    """
    print("--- Starting BPE Tokenizer Test ---")

    # 1. Create a small sample corpus
    sample_corpus = [
        "this is a simple sentence.",
        "here is another one.",
        "tokenizer test"
    ]
    print("\nSample Corpus created.")

    # 2. Initialize and train the tokenizer with a small vocab size for the test
    tokenizer = BpeTokenizer()
    tokenizer.train(sample_corpus, vocab_size=300)
    tokenizer.get_vocab_size() # Build the vocab
    tokenizer.special_tokens_map_inv = {i: s for s, i in tokenizer.vocab.items() if s in tokenizer.special_tokens}
    print(f"\nTokenizer trained. Total merges: {len(tokenizer.merges)}")

    # 3. Test tokenization (encoding)
    test_sentence = "this is a sentence"
    token_ids = tokenizer.encode(test_sentence)
    print(f"\nOriginal Sentence: '{test_sentence}'")
    print(f"Encoded IDs: {token_ids}")

    # 4. Test decoding
    decoded_text = tokenizer.decode(token_ids)
    print(f"Decoded Text: '{decoded_text}'")
    assert test_sentence == decoded_text, f"Decode mismatch: Got '{decoded_text}', Expected '{test_sentence}'"
    print("  Encode/Decode successful!")

    # 6. Test save and load functionality
    filepath = "test_tokenizer.json"
    tokenizer.save(filepath)
    print(f"\nTokenizer saved to '{filepath}'")

    # Load the saved state into a new tokenizer instance
    new_tokenizer = BpeTokenizer()
    new_tokenizer.load(filepath)
    print(f"New tokenizer loaded from '{filepath}'")

    # Verify that the loaded tokenizer works the same
    re_encoded_ids = new_tokenizer.encode(test_sentence)
    print(f"\nRe-encoded IDs with loaded tokenizer: {re_encoded_ids}")
    assert token_ids == re_encoded_ids
    print("  Save/Load functionality verified!")
    
    # Clean up the test file
    os.remove(filepath)
    print(f"Cleaned up '{filepath}'")

    print("\n--- BPE Tokenizer Test Passed Successfully! ---")

if __name__ == "__main__":
    test_tokenizer()
