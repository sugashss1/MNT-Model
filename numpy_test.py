# --- GPU/Device Setup ---
try:
    import cupy as np
    np.cuda.runtime.getDeviceCount() # Check for GPU
    print("--- Using CuPy for numpy_test.py ---")
except (ImportError, np.cuda.runtime.CUDARuntimeError):
    import numpy as np
    print("--- Using NumPy for numpy_test.py ---")

from autograd import Tensor
from numpy_layers import (
    MultiHeadAttention, 
    FeedForward, 
    PositionalEncoding, 
    EncoderLayer, 
    DecoderLayer, 
    Transformer
)

# --- Test Functions for NumPy-based Transformer ---

def test_multihead_attention_shapes():
    print("Testing MultiHeadAttention...")
    batch_size, seq_len, d_model, num_heads = 2, 5, 16, 4
    mha = MultiHeadAttention(d_model, num_heads)

    # Use our autograd Tensor with NumPy data
    Q = Tensor(np.random.randn(batch_size, seq_len, d_model))
    K = Tensor(np.random.randn(batch_size, seq_len, d_model))
    V = Tensor(np.random.randn(batch_size, seq_len, d_model))

    output = mha.forward(Q, K, V)
    
    expected_shape = (batch_size, seq_len, d_model)
    assert output.shape == expected_shape, f"Failed: Expected {expected_shape}, Got {output.shape}"
    print("  Passed!")

def test_feedforward_shapes():
    print("Testing FeedForward...")
    batch_size, seq_len, d_model, d_ff = 2, 6, 32, 64
    ff = FeedForward(d_model, d_ff)

    x = Tensor(np.random.randn(batch_size, seq_len, d_model))
    output = ff.forward(x)

    expected_shape = (batch_size, seq_len, d_model)
    assert output.shape == expected_shape, f"Failed: Expected {expected_shape}, Got {output.shape}"
    print("  Passed!")

def test_positional_encoding_adds_signal():
    print("Testing PositionalEncoding...")
    batch_size, seq_len, d_model = 2, 10, 32
    pe = PositionalEncoding(d_model, max_seq_length=50)

    x = Tensor(np.zeros((batch_size, seq_len, d_model)))
    out = pe.forward(x)

    # The output should not be all zeros
    assert not np.allclose(out.data(), np.zeros_like(out.data())), "Failed: Positional encoding did not add a signal."
    print("  Passed!")

def test_encoder_layer_shapes():
    print("Testing EncoderLayer...")
    batch_size, seq_len, d_model, num_heads, d_ff = 2, 7, 32, 4, 64
    enc_layer = EncoderLayer(d_model, num_heads, d_ff, dropout_p=0.1)

    x = Tensor(np.random.randn(batch_size, seq_len, d_model))
    # A simple mask (all ones, meaning no tokens are masked)
    mask = Tensor(np.ones((batch_size, 1, 1, seq_len)))

    output = enc_layer.forward(x, mask)
    
    expected_shape = (batch_size, seq_len, d_model)
    assert output.shape == expected_shape, f"Failed: Expected {expected_shape}, Got {output.shape}"
    print("  Passed!")

def test_decoder_layer_shapes():
    print("Testing DecoderLayer...")
    batch_size, seq_len, d_model, num_heads, d_ff = 2, 7, 32, 4, 64
    dec_layer = DecoderLayer(d_model, num_heads, d_ff, dropout_p=0.1)

    x = Tensor(np.random.randn(batch_size, seq_len, d_model))
    enc_output = Tensor(np.random.randn(batch_size, seq_len, d_model))
    src_mask = Tensor(np.ones((batch_size, 1, 1, seq_len)))
    tgt_mask = Tensor(np.ones((batch_size, 1, seq_len, seq_len)))

    output = dec_layer.forward(x, enc_output, src_mask, tgt_mask)
    
    expected_shape = (batch_size, seq_len, d_model)
    assert output.shape == expected_shape, f"Failed: Expected {expected_shape}, Got {output.shape}"
    print("  Passed!")

def test_transformer_forward_shapes():
    print("Testing full Transformer forward pass...")
    batch_size, src_len, tgt_len = 2, 8, 6
    src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff = 100, 120, 32, 4, 2, 64

    model = Transformer(
        src_vocab_size, tgt_vocab_size,
        d_model, num_heads, num_layers,
        d_ff, max_seq_length=50, dropout_p=0.1
    )

    # Use NumPy to generate random integer tensors
    src_data = np.random.randint(0, src_vocab_size, (batch_size, src_len))
    tgt_data = np.random.randint(0, tgt_vocab_size, (batch_size, tgt_len))
    src = Tensor(src_data)
    tgt = Tensor(tgt_data)

    output = model.forward(src, tgt)
    
    expected_shape = (batch_size, tgt_len, tgt_vocab_size)
    assert output.shape == expected_shape, f"Failed: Expected {expected_shape}, Got {output.shape}"
    print("  Passed!")


if __name__ == "__main__":
    test_multihead_attention_shapes()
    test_feedforward_shapes()
    test_positional_encoding_adds_signal()
    test_encoder_layer_shapes()
    test_decoder_layer_shapes()
    test_transformer_forward_shapes()
    print("\nAll NumPy layer tests passed successfully!")
