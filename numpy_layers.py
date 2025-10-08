# --- GPU/Device Setup ---
try:
    import cupy as np
    np.cuda.runtime.getDeviceCount() # Check for GPU
    IS_GPU = True
except (ImportError, np.cuda.runtime.CUDARuntimeError):
    import numpy as np
    IS_GPU = False

# We'll be using the Tensor class you provided
from autograd import Tensor, array, Exp

class ReLU:
    """
    Applies the Rectified Linear Unit function element-wise.
    
    This is implemented as a class with forward and backward methods
    to integrate with the custom autograd engine.
    """
    def forward(self, a):
        requires_grad = a.requires_grad
        
        # Apply ReLU function
        data = np.maximum(0, a._data)
        
        # Create output Tensor
        z = Tensor(data, requires_grad=requires_grad, operation=self)
        
        # Build the computation graph
        self.parents = (a,)
        a.children.append(z)
        self.cache = a # Cache input tensor for backprop

        return z

    def backward(self, dz, z):
        a = self.cache
        
        if a.requires_grad:
            # The gradient of ReLU is 1 for positive inputs and 0 otherwise.
            # We multiply the incoming gradient `dz` by this mask.
            da = dz * (a._data > 0)
            a.backward(da, z)

class Linear:
    """
    A fully connected linear layer, similar to torch.nn.Linear.

    This layer applies a linear transformation to the incoming data: y = xA^T + b.
    It uses the custom Tensor class from autograd.py for automatic differentiation.
    """
    def __init__(self, in_features, out_features, bias=True):
        """
        Initializes the layer with weights and biases.

        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features.
            bias (bool): If True, adds a learnable bias to the output.
        """
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias

        # Initialize weights with Xavier/Glorot initialization for better training stability
        # This helps prevent gradients from vanishing or exploding.
        limit = np.sqrt(6.0 / (in_features + out_features))
        weights_data = np.random.uniform(-limit, limit, (out_features, in_features)).astype(np.float32)
        self.weights = Tensor(weights_data, requires_grad=True)

        if self.use_bias:
            # Initialize biases to zero
            bias_data = np.zeros(out_features, dtype=np.float32)
            self.bias = Tensor(bias_data, requires_grad=True)

    def forward(self, x):
        """
        Performs the forward pass of the linear layer.

        Args:
            x (Tensor): The input tensor of shape (..., in_features).

        Returns:
            Tensor: The output tensor of shape (..., out_features).
        """
        # Perform the matrix multiplication: input @ weights_transposed
        output = x @ self.weights.transpose(-1, -2)
        
        if self.use_bias:
            # Add the bias term
            output += self.bias
            
        return output

    def __repr__(self):
        return f"Linear(in_features={self.in_features}, out_features={self.out_features}, bias={self.use_bias})"

class FeedForward:
    """
    Implements the Position-wise Feed-Forward Network from the Transformer paper.
    This consists of two linear transformations with a ReLU activation in between.
    """
    def __init__(self, d_model, d_ff):
        """
        Initializes the FeedForward network.

        Args:
            d_model (int): The dimensionality of the input and output.
            d_ff (int): The dimensionality of the inner-layer.
        """
        self.d_model = d_model
        self.d_ff = d_ff
        
        self.fc1 = Linear(d_model, d_ff)
        self.fc2 = Linear(d_ff, d_model)
        self.relu = ReLU()

    def forward(self, x):
        """
        Performs the forward pass of the network.

        Args:
            x (Tensor): The input tensor of shape (batch_size, seq_len, d_model).

        Returns:
            Tensor: The output tensor of shape (batch_size, seq_len, d_model).
        """
        # Pass through the first linear layer
        x = self.fc1.forward(x)
        # Apply ReLU activation
        x = self.relu.forward(x)
        # Pass through the second linear layer
        x = self.fc2.forward(x)
        return x
    
    def __repr__(self):
        return f"FeedForward(d_model={self.d_model}, d_ff={self.d_ff})"

def softmax(x, dim=-1):
    """
    Computes the softmax function along a given dimension.
    
    This function is built using operations from the custom autograd engine to ensure
    that backpropagation works correctly. It includes a trick for numerical stability.
    """
    # Subtract the max for numerical stability (prevents overflow when exponentiating)
    x_max = x.max(dim=dim, keepdims=True)
    e_x = Exp().forward(x - x_max)
    sum_e_x = e_x.sum(dim=dim, keepdims=True)
    return e_x / sum_e_x

class MultiHeadAttention:
    """
    Implements the Multi-Head Attention mechanism from the Transformer paper.
    """
    def __init__(self, d_model, num_heads):
        """
        Initializes the MultiHeadAttention layer.

        Args:
            d_model (int): The dimensionality of the input embeddings.
            num_heads (int): The number of attention heads.
        """
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads # Dimensionality of each head's key/query/value

        self.W_q = Linear(d_model, d_model)
        self.W_k = Linear(d_model, d_model)
        self.W_v = Linear(d_model, d_model)
        self.W_o = Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """
        Calculates the scaled dot-product attention scores.

        Args:
            Q (Tensor): Query tensor.
            K (Tensor): Key tensor.
            V (Tensor): Value tensor.
            mask (Tensor, optional): A mask to prevent attention to certain positions.

        Returns:
            Tensor: The output of the attention mechanism.
        """
        # Use np.sqrt on a float, which works for both numpy and cupy
        # Or ensure self.d_k is a cupy array if needed, but this is safer.
        scaling_factor = np.sqrt(np.array(self.d_k, dtype=np.float32))
        attn_scores = (Q @ K.transpose(-2, -1)) / scaling_factor
        
        if mask is not None:
            # Where mask is 0, fill with a large negative number to make softmax output close to 0
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        attn_probs = softmax(attn_scores, dim=-1)
        output = attn_probs @ V
        return output

    def split_heads(self, x):
        """
        Splits the last dimension of the input tensor into multiple heads.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, d_model).

        Returns:
            Tensor: Reshaped tensor of shape (batch_size, num_heads, seq_len, d_k).
        """
        batch_size, seq_len, d_model = x.shape
        return x.reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        """
        Combines the multiple heads back into a single tensor.

        Args:
            x (Tensor): Input tensor of shape (batch_size, num_heads, seq_len, d_k).

        Returns:
            Tensor: Combined tensor of shape (batch_size, seq_len, d_model).
        """
        batch_size, num_heads, seq_len, d_k = x.shape
        return x.transpose(1, 2).reshape(batch_size, seq_len, self.d_model)

    def forward(self, Q, K, V, mask=None):
        """
        Performs the forward pass of the MultiHeadAttention layer.
        """
        # 1. Linearly project Q, K, V and split into heads
        Q = self.split_heads(self.W_q.forward(Q))
        K = self.split_heads(self.W_k.forward(K))
        V = self.split_heads(self.W_v.forward(V))

        # 2. Apply scaled dot-product attention
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)

        # 3. Combine heads and apply final linear projection
        output = self.W_o.forward(self.combine_heads(attn_output))
        return output

class LayerNorm:
    """
    Implements Layer Normalization.
    """
    def __init__(self, d_model, eps=1e-5):
        """
        Initializes the LayerNorm module.

        Args:
            d_model (int): The dimensionality of the input.
            eps (float): A small value added to the denominator for numerical stability.
        """
        self.d_model = d_model
        self.eps = eps
        # Gamma: learnable scaling parameter, initialized to ones
        self.gamma = Tensor(np.ones(d_model, dtype=np.float32), requires_grad=True)
        # Beta: learnable shift parameter, initialized to zeros
        self.beta = Tensor(np.zeros(d_model, dtype=np.float32), requires_grad=True)

    def forward(self, x):
        """
        Performs the forward pass of the LayerNorm.
        """
        mean = x.mean(dim=-1, keepdims=True)
        var = x.var(dim=-1, keepdims=True)
        
        # Normalize the input
        # FIX: Changed .sqrt() to ** 0.5, which is supported by autograd's Pow operation
        x_norm = (x - mean) / ((var + self.eps).sqrt())
        
        # Scale and shift
        return self.gamma * x_norm + self.beta

class Dropout:
    """
    Implements the Dropout layer.
    """
    def __init__(self, p=0.1):
        """
        Initializes the Dropout layer.

        Args:
            p (float): The probability of an element to be zeroed.
        """
        self.p = p
        self.train = True # Dropout is only active during training

    def forward(self, x):
        """
        Performs the forward pass of the Dropout layer.
        """
        if self.train:
            # Create a mask of 0s and 1s
            mask = np.random.binomial(1, 1.0 - self.p, size=x.shape)
            # Apply the mask and scale the output to keep the same expected value
            # We don't need a custom autograd operation for this, as the gradient
            # will just flow through the multiplication.
            return (x * mask) / (1.0 - self.p)
        return x
    
    def eval(self):
        """Sets the layer to evaluation mode."""
        self.train = False

    def train_mode(self):
        """Sets the layer to training mode."""
        self.train = True

class EncoderLayer:
    """
    Implements a single layer of the Transformer Encoder.
    
    It consists of a multi-head self-attention mechanism followed by a 
    position-wise feed-forward network. Residual connections and layer 
    normalization are applied after each sub-layer.
    """
    def __init__(self, d_model, num_heads, d_ff, dropout_p):
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout_p)
        self.dropout2 = Dropout(dropout_p)

    def forward(self, x, mask=None):
        """
        Performs the forward pass for the encoder layer.
        """
        # --- Self-Attention Sub-layer ---
        # 1. Calculate attention
        attn_output = self.self_attn.forward(x, x, x, mask)
        # 2. Apply dropout and the first residual connection, then normalize
        x = self.norm1.forward(x + self.dropout1.forward(attn_output))
        
        # --- Feed-Forward Sub-layer ---
        # 1. Pass through the feed-forward network
        ff_output = self.feed_forward.forward(x)
        # 2. Apply dropout and the second residual connection, then normalize
        x = self.norm2.forward(x + self.dropout2.forward(ff_output))
        
        return x

class DecoderLayer:
    """
    Implements a single layer of the Transformer Decoder.

    It consists of a masked multi-head self-attention, a multi-head 
    cross-attention over the encoder's output, and a position-wise 
    feed-forward network.
    """
    def __init__(self, d_model, num_heads, d_ff, dropout_p):
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout_p)
        self.dropout2 = Dropout(dropout_p)
        self.dropout3 = Dropout(dropout_p)

    def forward(self, x, enc_output, src_mask, tgt_mask):
        """
        Performs the forward pass for the decoder layer.
        """
        # --- Masked Self-Attention Sub-layer ---
        attn_output = self.self_attn.forward(x, x, x, tgt_mask)
        x = self.norm1.forward(x + self.dropout1.forward(attn_output))
        
        # --- Cross-Attention Sub-layer ---
        # Query comes from the decoder, Key and Value from the encoder's output
        attn_output = self.cross_attn.forward(x, enc_output, enc_output, src_mask)
        x = self.norm2.forward(x + self.dropout2.forward(attn_output))
        
        # --- Feed-Forward Sub-layer ---
        ff_output = self.feed_forward.forward(x)
        x = self.norm3.forward(x + self.dropout3.forward(ff_output))
        
        return x

class Embedding:
    """
    A simple embedding layer that turns positive integers (indices) into dense vectors of fixed size.
    """
    def __init__(self, num_embeddings, embedding_dim):
        """
        Initializes the Embedding layer.

        Args:
            num_embeddings (int): The size of the vocabulary.
            embedding_dim (int): The size of each embedding vector.
        """
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        
        # Initialize the embedding table with random values
        weight_data = np.random.randn(num_embeddings, embedding_dim).astype(np.float32)
        self.weight = Tensor(weight_data, requires_grad=True)

    def forward(self, x):
        """
        Performs the embedding lookup.

        Args:
            x (Tensor): A tensor of integer indices.

        Returns:
            Tensor: The corresponding embedding vectors.
        """
        # We can use the Slice operation from autograd to perform the lookup
        return self.weight[x.data()]

class PositionalEncoding:
    """
    Implements the Positional Encoding function.
    """
    def __init__(self, d_model, max_seq_length):
        """
        Initializes the PositionalEncoding module.

        Args:
            d_model (int): The dimensionality of the embeddings.
            max_seq_length (int): The maximum possible length of a sequence.
        """
        pe = np.zeros((max_seq_length, d_model))
        position = np.arange(0, max_seq_length, dtype=np.float32).reshape(-1, 1)
        div_term = np.exp(np.arange(0, d_model, 2).astype(np.float32) * -(np.log(10000.0) / d_model))
        
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        
        # Add a batch dimension and convert to a non-trainable Tensor
        self.pe = Tensor(pe.astype(np.float32), requires_grad=False).reshape(1, max_seq_length, d_model)

    def forward(self, x):
        """
        Adds positional encoding to the input tensor.
        """
        # Add the positional encoding up to the length of the input sequence
        return x + self.pe[:, :x.shape[1]]

class Transformer:
    """
    The main Transformer model, combining all the building blocks.
    """
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout_p):
        self.encoder_embedding = Embedding(src_vocab_size, d_model)
        self.decoder_embedding = Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)

        self.encoder_layers = [EncoderLayer(d_model, num_heads, d_ff, dropout_p) for _ in range(num_layers)]
        self.decoder_layers = [DecoderLayer(d_model, num_heads, d_ff, dropout_p) for _ in range(num_layers)]

        self.fc = Linear(d_model, tgt_vocab_size)
        self.dropout = Dropout(dropout_p)

    def generate_mask(self, src, tgt):
        # src_mask: prevent attention to padding tokens in the source
        src_mask = (src.data() != 0).reshape(src.shape[0], 1, 1, src.shape[1])
        
        # tgt_mask: prevent attention to padding tokens in the target
        tgt_mask = (tgt.data() != 0).reshape(tgt.shape[0], 1, 1, tgt.shape[1])
        
        # nopeak_mask: prevent attention to future tokens in the target (causal mask)
        seq_length = tgt.shape[1]
        nopeak_mask = np.triu(np.ones((1, seq_length, seq_length)), k=1).astype('bool')
        
        # Combine the padding mask and the causal mask for the target
        final_tgt_mask = tgt_mask & ~nopeak_mask
        
        return Tensor(src_mask), Tensor(final_tgt_mask)

    def forward(self, src, tgt):
        src_mask, tgt_mask = self.generate_mask(src, tgt)

        src_embedded = self.dropout.forward(self.positional_encoding.forward(self.encoder_embedding.forward(src)))
        tgt_embedded = self.dropout.forward(self.positional_encoding.forward(self.decoder_embedding.forward(tgt)))

        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer.forward(enc_output, src_mask)

        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer.forward(dec_output, enc_output, src_mask, tgt_mask)

        output = self.fc.forward(dec_output)
        return output
