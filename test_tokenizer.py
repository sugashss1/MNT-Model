from bpe_tokenizer import BpeTokenizer
import os

def test_tokenizer():
    """
    A comprehensive test for the BpeTokenizer.
    """
    print("--- Starting BPE Tokenizer Test ---")

    # 1. Create a small sample corpus with English and Tamil
    sample_corpus = [
        "this is a simple sentence.",
        "here is another one.",
        "இது ஒரு எளிய வாக்கியம்.",
        "இதோ இன்னொன்று.",
        "machine translation is fun.",
        "இயந்திர மொழிபெயர்ப்பு வேடிக்கையாக உள்ளது."
    ]
    print("\nSample Corpus created.")

    # 2. Initialize and train the tokenizer with a small vocab size for the test
    # A larger vocab size (e.g., 10000) would be used for the real corpus.
    tokenizer = BpeTokenizer(vocab_size=100)
    tokenizer.train(sample_corpus)

    print(f"\nLearned Vocabulary (first 20 items): {dict(list(tokenizer.vocab.items())[:20])}...")
    print(f"Learned Merges (first 10): {dict(list(tokenizer.merges.items())[:10])}...")

    # 3. Test tokenization (encoding)
    test_sentence_en = "this is a fun sentence"
    token_ids_en = tokenizer.tokenize(test_sentence_en)
    print(f"\nOriginal English Sentence: '{test_sentence_en}'")
    print(f"Tokenized IDs: {token_ids_en}")

    # 4. Test decoding
    decoded_text_en = tokenizer.decode(token_ids_en)
    print(f"Decoded Text: '{decoded_text_en}'")
    assert test_sentence_en == decoded_text_en
    print("  English Encode/Decode successful!")

    # 5. Test with a Tamil sentence
    test_sentence_ta = "இது ஒரு வாக்கியம்"
    token_ids_ta = tokenizer.tokenize(test_sentence_ta)
    print(f"\nOriginal Tamil Sentence: '{test_sentence_ta}'")
    print(f"Tokenized IDs: {token_ids_ta}")
    decoded_text_ta = tokenizer.decode(token_ids_ta)
    print(f"Decoded Text: '{decoded_text_ta}'")
    assert test_sentence_ta == decoded_text_ta
    print("  Tamil Encode/Decode successful!")

    # 6. Test save and load functionality
    filepath = "test_tokenizer.json"
    tokenizer.save(filepath)
    print(f"\nTokenizer saved to '{filepath}'")

    # Create a new tokenizer instance and load the saved state
    new_tokenizer = BpeTokenizer()
    new_tokenizer.load(filepath)
    print(f"New tokenizer loaded from '{filepath}'")

    # Verify that the loaded tokenizer works the same
    retokenized_ids = new_tokenizer.tokenize(test_sentence_en)
    print(f"\nRetokenized IDs with loaded tokenizer: {retokenized_ids}")
    assert token_ids_en == retokenized_ids
    print("  Save/Load functionality verified!")
    
    # Clean up the test file
    os.remove(filepath)
    print(f"Cleaned up '{filepath}'")

    print("\n--- BPE Tokenizer Test Passed Successfully! ---")

if __name__ == "__main__":
    test_tokenizer()
