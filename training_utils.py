# --- GPU/Device Setup ---
try:
    import cupy as np
    np.cuda.runtime.getDeviceCount() # Check for GPU
    IS_GPU = True
except (ImportError, np.cuda.runtime.CUDARuntimeError):
    import numpy as np
    IS_GPU = False

from autograd import Tensor, tensor

class CrossEntropyLoss:
    """
    Computes the Cross-Entropy Loss between model outputs and target labels.
    This version is numerically stable.
    """
    def forward(self, model_output: Tensor, targets: np.ndarray) -> Tensor:
        """
        Args:
            model_output (Tensor): The raw logits from the model. 
                                   Can be 2D (batch_size * seq_len, vocab_size) or
                                   3D (batch_size, seq_len, vocab_size).
            targets (np.ndarray): The ground truth labels. 
                                  Can be 1D (batch_size * seq_len,) or
                                  2D (batch_size, seq_len).
        """
        # If inputs are 3D/2D, reshape them to 2D/1D
        if len(model_output.shape) == 3:
            batch_size, seq_len, vocab_size = model_output.shape
            model_output = model_output.reshape(batch_size * seq_len, vocab_size)
            targets = targets.flatten()

        # --- Numerically Stable Log-Softmax ---
        # 1. Subtract the max for stability
        max_val = model_output.max(dim=-1, keepdims=True)
        # The subtraction here uses broadcasting
        log_softmax_output = model_output - max_val - (model_output - max_val).exp().sum(dim=-1, keepdims=True).log()
        
        # --- Negative Log Likelihood ---
        # 2. Gather the log probabilities of the correct classes
        num_samples = log_softmax_output.shape[0]
        # We use np.arange and the targets array to select the correct log probability for each sample
        correct_log_probs = log_softmax_output[np.arange(num_samples), targets]

        # 3. Compute the mean of the negative log probabilities
        loss = -correct_log_probs.mean()
        
        return loss

class Adam:
    """ Implements the Adam optimizer. """
    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.params = [p for p in params if p.requires_grad]
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        
        self.m = [np.zeros_like(p.data()) for p in self.params]
        self.v = [np.zeros_like(p.data()) for p in self.params]

    def step(self):
        """ Performs a single optimization step. """
        self.t += 1
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue

            # Update biased first moment estimate
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * param.grad
            
            # Update biased second raw moment estimate
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (param.grad ** 2)
            
            # Compute bias-corrected first moment estimate
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            
            # Compute bias-corrected second raw moment estimate
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            
            # Update parameters
            # Use np.sqrt which will be cupy.sqrt if available
            update_val = self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
            param._data -= update_val
            if IS_GPU:
                np.cuda.Stream.null.synchronize()

    def zero_grad(self):
        """ Deprecated. Use the zero_all_grads helper in train.py instead. """
        for p in self.params:
            p.zero_grad()
