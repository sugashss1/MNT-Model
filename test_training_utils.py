import numpy as np
import torch
from autograd import Tensor
from training_utils import CrossEntropyLoss, Adam

def test_cross_entropy_loss():
    """
    Tests our custom CrossEntropyLoss against PyTorch's implementation
    to verify correctness of both forward and backward passes.
    """
    print("--- Testing CrossEntropyLoss ---")
    
    # 1. Setup parameters
    batch_size, seq_len, vocab_size = 2, 5, 10
    
    # 2. Create identical random data for both frameworks
    np_logits = np.random.randn(batch_size, seq_len, vocab_size).astype(np.float32)
    np_targets = np.random.randint(0, vocab_size, (batch_size, seq_len))

    # PyTorch setup
    torch_logits = torch.tensor(np_logits, requires_grad=True)
    torch_targets = torch.tensor(np_targets, dtype=torch.long)
    torch_loss_fn = torch.nn.CrossEntropyLoss()
    
    # Our framework setup
    our_logits = Tensor(np_logits, requires_grad=True)
    our_loss_fn = CrossEntropyLoss()

    # 3. Forward pass comparison
    # PyTorch requires reshaping for this loss function
    torch_loss = torch_loss_fn(torch_logits.view(-1, vocab_size), torch_targets.view(-1))
    our_loss = our_loss_fn.forward(our_logits, np_targets)

    print(f"  Our loss: {our_loss.data().item():.6f}")
    print(f"  PyTorch loss: {torch_loss.item():.6f}")
    assert np.allclose(our_loss.data(), torch_loss.detach().numpy()), "Forward pass values do not match!"
    print("  Forward pass: Passed!")

    # 4. Backward pass comparison
    torch_loss.backward()
    our_loss.backward()

    assert np.allclose(our_logits.grad, torch_logits.grad.numpy(), atol=1e-6), "Backward pass gradients do not match!"
    print("  Backward pass: Passed!")
    print("--- CrossEntropyLoss Test Passed! ---")


def test_adam_optimizer():
    """
    Tests the Adam optimizer to ensure it updates parameters correctly.
    """
    print("\n--- Testing Adam Optimizer ---")

    # 1. Create simple parameters for a "dummy" model
    w = Tensor(np.random.randn(2, 3), requires_grad=True)
    b = Tensor(np.random.randn(1, 3), requires_grad=True)
    
    # Store initial values to check for updates
    w_initial = w.data().copy()
    b_initial = b.data().copy()
    
    # 2. Initialize the optimizer with the parameters
    optimizer = Adam(params=[w, b], lr=0.01)

    # 3. Simulate a forward and backward pass
    x = Tensor(np.random.randn(4, 2))
    # Dummy loss calculation
    loss = (x @ w + b).sum() 
    loss.backward()
    
    assert w.grad is not None and np.any(w.grad != 0), "Gradient for 'w' was not computed."
    assert b.grad is not None and np.any(b.grad != 0), "Gradient for 'b' was not computed."
    print("  Gradients successfully computed.")

    # 4. Perform an optimization step
    optimizer.step()
    print("  Optimizer step completed.")
    
    # 5. Check if parameters have been updated
    assert not np.allclose(w.data(), w_initial), "Weight 'w' was not updated by the optimizer."
    assert not np.allclose(b.data(), b_initial), "Bias 'b' was not updated by the optimizer."
    print("  Parameters successfully updated.")

    # 6. Check if gradients are zeroed out
    optimizer.zero_grad()
    # After zero_grad_tree, the grad attribute should be a new array of zeros
    assert np.all(w.grad == 0), "Gradients for 'w' were not zeroed."
    assert np.all(b.grad == 0), "Gradients for 'b' were not zeroed."
    print("  Gradients successfully zeroed.")
    print("--- Adam Optimizer Test Passed! ---")

if __name__ == "__main__":
    test_cross_entropy_loss()
    test_adam_optimizer()
