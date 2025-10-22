from typing import List

# --- GPU/Device Setup ---
try:
    import cupy as np
    np.cuda.runtime.getDeviceCount() # Check for GPU
    IS_GPU = True
except (ImportError, np.cuda.runtime.CUDARuntimeError):
    import numpy as np
    IS_GPU = False

# Tensor class, with __init__, backward, magic methods, and utils:
class Tensor:
    ''' Tensor class, with __init__, backward, magic methods, and utils '''
    def __init__(self, data, requires_grad = False, operation = None) -> None:
        '''
        Creates new instance of the Tensor class.

        @param data (Array-like): Iterable containing the data to be stored in the Tensor.
        @param requires_grad (Bool): Whether to keep track of the Tensor's gradients.
        @param operation (Operation Object): When a tensor is created from other tensors, this stores
        the operation that generated the new tensor (e.g. "Add", "Exp", "MatMul").
        '''
        self._data = array(data)
        self.requires_grad = requires_grad
        self.operation = operation
        self.children = []
        self.shape = self._data.shape
        self.grad = None # Lazy initialization of gradient

    def __repr__(self):
        return f"""({self._data}, requires_grad = {self.requires_grad})"""

    def data(self):
        ''' Returns the data stored in the tensor as a Numpy Array. '''
        return self._data
    
    def backward(self, grad = None, z = None):
        ''' 
        Performs the backpropagation with gradient descent from current tensor.
        Will fill every tensor's "grad" attribute with gradients relative to "self" (current Tensor).
        '''
        if not self.requires_grad:
            return "this tensor has requires_grad set to False"
        
        # Initialize the top-level gradient if it's the start of the chain
        if grad is None:
            grad = np.ones_like(self._data, dtype=np.float32)

        # Accumulate gradients
        if self.grad is None:
            self.grad = grad
        else:
            self.grad += grad

        # --- Memory Optimization ---
        # If this backward call came from a child tensor `z`, we can try to free memory.
        if z is not None:
            # Remove the reference from the child to this parent, as its contribution is done.
            if z in self.children:
                self.children.remove(z)
        
        # --- Graph Traversal ---
        # Only propagate backwards if all children have backpropagated to this node.
        # This ensures the gradient is fully accumulated before passing it to parents.
        if self.operation:
            if not self.children:
                self.operation.backward(self.grad, self)

    def tolist(self):
        ''' Turns the Tensor into a python list. '''
        return self._data.tolist()

    def toarray(self):
        ''' Turns the Tensor into a numpy array. '''
        return self._data
    
    def zero_grad(self):
        ''' Reset the Tensor's gradients to zero. '''
        self.grad = None # Resetting to None frees the memory

    def zero_grad_tree(self):
        ''' Reset the gradients of this Tensor, and of all of the Tensors that led to it. '''
        self.zero_grad()
        if self.operation:
            for parent in self.operation.parents:
                parent.zero_grad_tree()
            self.operation = None

    def __add__(self, other):
        """ New = self + other """
        op = Add()
        return op.forward(self, tensor(other))

    def __radd__(self, other):
        """ New = other + self """
        op = Add()
        return op.forward(self, tensor(other))

    def __iadd__(self, other):
        """ self += other """
        op = Add()
        return op.forward(self, tensor(other))

    def __sub__(self, other):
        """ New = self - other """
        op=Sub()
        return op.forward( self , tensor(other))

    def __rsub__(self, other):
        """ New = other - self """
        op=Sub()
        return op.forward( tensor(other) , self)

    def __isub__(self, other):
        """ self -= other """
        op=Sub()
        return op.forward( self , tensor(other))
    
    def __neg__(self):
        """ self = -self """
        op = Neg()
        return op.forward(self) 

    def __mul__(self, other):
        """ New = self * other """
        op = Mul()
        return op.forward(self, tensor(other))

    def __rmul__(self, other):
        """ New = other * self """
        op = Mul()
        return op.forward(self, tensor(other))

    def __imul__(self, other):
        """ self *= other """
        op = Mul()
        return op.forward(self, tensor(other))
    
    def __pow__(self, other):
        op = Pow()
        return op.forward(self, tensor(other))

    def __matmul__(self, other):
        """ New = self @ other """
        op = MatMul()
        return op.forward(self, tensor(other))
    
    def __truediv__(self, other):
        """ New = self / other """
        op = Div()
        return op.forward(self, tensor(other))
    
    def __getitem__(self, index): 
        """ New = self[index] """
        op = Slice()
        return op.forward(self, index)

    def __gt__(self, other):
        """ New = self > other """
        return self._data > array(other)

    def max(self, dim=-1, keepdims=False):
        """
        Returns the largest values across the "dim" dimention.
        Example: (B, T, D), dim = 1 -> (B, D).
        
        @param dim (int): dimention to be reduced (only largest remains).
        @param keepdims (bool): wether to broadcast result to same shape as input.
        """
        op = Max()
        return op.forward(self, dim, keepdims=keepdims)

    def sum(self, dim=-1, keepdims=False):
        """
        Returns the sum of all values across the "dim" dimention.
        Example: (B, T, D), dim = 1 -> (B, D).
        
        @param dim (int): dimention to be summed across.
        @param keepdims (bool): wether to broadcast result to same shape as input.
        """
        op = Sum()
        return op.forward(self, dim, keepdims=keepdims)

    def mean(self, dim=-1, keepdims=False):
        """
        Returns the mean of all values across the "dim" dimention.
        Example: (B, T, D), dim = 1 -> (B, D).

        @param dim (int): dimention to be averaged across.
        @param keepdims (bool): wether to broadcast result to same shape as input.
        """
        op = Mean()
        return op.forward(self, dim, keepdims=keepdims)

    def var(self, dim=-1, keepdims=False):
        """
        Returns the variance of all values across the "dim" dimention.
        Example: (B, T, D), dim = 1 -> (B, D).

        @param dim (int): dimention the variance will be computed across.
        @param keepdims (bool): wether to broadcast result to same shape as input.
        """
        op = Var()
        return op.forward(self, dim, keepdims=keepdims)

    def exp(self):
        """ Returns e^self """
        op = Exp()
        return op.forward(self)

    def log(self):
        """ Returns the natural logarithm of self """
        op = Log()
        return op.forward(self)

    def sqrt(self):
        """ Returns the square root of self """
        op = Sqrt()
        return op.forward(self)

    def reshape(self, *shape):
        """
        Returns the original tensor reshaped to the new shape given.
        Example: (16, 8, 4), *shape =(2, 32, 8) -> (2, 32, 8)

        @param *shape (integers): new shape of the tensor.
        """
        op = Reshape()
        return op.forward(self, shape)

    def transpose(self, *dims):
        """
        Returns the original tensor with the two given dimentions transposed.
        Example: (16, 8, 4), *dims=(-2,-1) -> (16, 4, 8)

        @param *dims (integers): two dimentions to be transposed.
        """
        op = Transpose()
        return op.forward(self, *dims)

    def masked_fill(self, condition, value):
        """
        Returns the original tensor with the values where condition is True set to "value".

        @param condition (Array-like): matrix with True and False. Where this is False, will replace original with value.
        @param value (float): value to fill Tensor with, where condition is True.
        """
        op = MaskedFill()
        return op.forward(self, array(condition), value )

        
# Operations between two tensors:
class Add:

    def forward(self, a, b):
        requires_grad = a.requires_grad or b.requires_grad
      
        # Get new Tensor's data:
        data = a._data + b._data
      
        # Create new Tensor:
        z = Tensor(data, requires_grad=requires_grad, operation=self) 
      
        # Add new Tensors to "children" and old Tensors to "parents":
        self.parents = (a, b)
        a.children.append(z)
        b.children.append(z)
        self.cache = (a, b)

        return z
    
    def backward(self, dz, z):
        a, b = self.cache
        
        # Find gradients relative to "a", and pass it downstream:
        if a.requires_grad:
            da = dz

            # Rescale gradient to have the same shape as "a":
            grad_dim = len(dz.shape)
            in_dim = len(a.shape)
            for _ in range(grad_dim - in_dim):
                da = da.sum(axis=0)
            
            for n, dim in enumerate(a.shape):
                if dim == 1:
                    da = da.sum(axis=n, keepdims=True)
            a.backward(da, z)

        # Find gradients relative to "b", and pass it downstream:
        if b.requires_grad:
            db = dz

            # Rescale gradient to have the same shape as "b":
            grad_dim = len(dz.shape)
            in_dim = len(b.shape)
            for _ in range(grad_dim - in_dim):
                db = db.sum(axis=0)

            for n, dim in enumerate(b.shape):
                if dim == 1:
                    db = db.sum(axis=n, keepdims=True)
            b.backward(db, z)

class Sub:

    def forward(self, a, b):
        requires_grad = a.requires_grad or b.requires_grad
      
        # Get new Tensor's data:
        data = a._data - b._data
      
        # Create new Tensor:
        z = Tensor(data, requires_grad=requires_grad, operation=self) 
      
        # Add new Tensors to "children" and old Tensors to "parents":
        self.parents = (a, b)
        a.children.append(z)
        b.children.append(z)
        self.cache = (a, b)

        return z
    
    def backward(self, dz, z):
        a, b = self.cache
        
        # Find gradients relative to "a", and pass it downstream:
        if a.requires_grad:
            da = dz

            # Rescale gradient to have the same shape as "a":
            grad_dim = len(dz.shape)
            in_dim = len(a.shape)
            for _ in range(grad_dim - in_dim):
                da = da.sum(axis=0)
            
            for n, dim in enumerate(a.shape):
                if dim == 1:
                    da = da.sum(axis=n, keepdims=True)
            a.backward(da, z)

        # Find gradients relative to "b", and pass it downstream:
        if b.requires_grad:
            db = -dz

            # Rescale gradient to have the same shape as "b":
            grad_dim = len(dz.shape)
            in_dim = len(b.shape)
            for _ in range(grad_dim - in_dim):
                db = db.sum(axis=0)

            for n, dim in enumerate(b.shape):
                if dim == 1:
                    db = db.sum(axis=n, keepdims=True)
            b.backward(db, z)


class Neg:

    def forward(self, a):
        requires_grad = a.requires_grad
   
        # Get new Tensor's data:
        data = - a._data 
   
        # Create new Tensor:
        z = Tensor(data, requires_grad=requires_grad, operation=self) 
   
        # Add new Tensors to "children" and old Tensors to "parents":
        self.parents = (a,)
        a.children.append(z)

        self.cache = a

        return z 
    
    def backward(self, dz, z):
        a = self.cache

        # Find gradients relative to "a", and pass it downstream:
        if a.requires_grad:
            da = -dz
            a.backward(da, z)


class Mul:

    def forward(self, a, b):
        requires_grad = a.requires_grad or b.requires_grad
       
        # Get new Tensor's data:
        data = a._data * b._data
       
        # Create new Tensor:
        z = Tensor(data, requires_grad=requires_grad, operation=self) 
       
        # Add new Tensors to "children" and old Tensors to "parents":
        self.parents = (a, b)
        a.children.append(z)
        b.children.append(z)
        self.cache = (a, b)

        return z 
    
    def backward(self, dz, z):
        a, b = self.cache
        
        # Find gradients relative to "a", and pass it downstream:
        if a.requires_grad:
            # d/da(a*b) = b, apply chain rule:
            da = dz * b._data

            # Rescale gradient to have the same shape as "a":
            grad_dim = len(dz.shape)
            in_dim = len(a.shape)
            for _ in range(grad_dim - in_dim):
                da = da.sum(axis=0)
            
            for n, dim in enumerate(a.shape):
                if dim == 1:
                    da = da.sum(axis=n, keepdims=True)
            a.backward(da, z)

        # Find gradients relative to "b", and pass it downstream:
        if b.requires_grad:
            # d/db(a*b) = a, apply chain rule:
            db = dz * a._data

            # Rescale gradient to have the same shape as "b":
            grad_dim = len(dz.shape)
            in_dim = len(b.shape)
            for _ in range(grad_dim - in_dim):
                db = db.sum(axis=0)

            for n, dim in enumerate(b.shape):
                if dim == 1:
                    db = db.sum(axis=n, keepdims=True)
            b.backward(db, z)


class Div:

    def forward(self, a, b):
        requires_grad = a.requires_grad or b.requires_grad
       
        # Get new Tensor's data:
        data = a._data / b._data
       
        # Create new Tensor:
        z = Tensor(data, requires_grad=requires_grad, operation=self) 
       
        # Add new Tensors to "children" and old Tensors to "parents":
        self.parents = (a, b)
        a.children.append(z)
        b.children.append(z)
        self.cache = (a, b)

        return z  

    
    def backward(self, dz, z):
        a, b = self.cache
        
        # Find gradients relative to "a", and pass it downstream:
        if a.requires_grad:
            # d/da(a/b) = (1/b), apply chain rule:
            da = dz * (1 / b._data)

            # Rescale gradient to have the same shape as "a":
            grad_dim = len(dz.shape)
            in_dim = len(a.shape)
            for _ in range(grad_dim - in_dim):
                da = da.sum(axis=0)
            
            for n, dim in enumerate(a.shape):
                if dim == 1:
                    da = da.sum(axis=n, keepdims=True)
            a.backward(da, z)

        # Find gradients relative to "b", and pass it downstream:
        if b.requires_grad:
            # d/db(a/b) = -(a/b^2), apply chain rule:
            db = - dz * a._data / (b._data ** 2)

            # Rescale gradient to have the same shape as "b":
            grad_dim = len(dz.shape)
            in_dim = len(b.shape)
            for _ in range(grad_dim - in_dim):
                db = db.sum(axis=0)

            for n, dim in enumerate(b.shape):
                if dim == 1:
                    db = db.sum(axis=n, keepdims=True)
            b.backward(db, z)

class Pow():
    def forward(self, tensor_a, tensor_b):
        requires_grad = tensor_a.requires_grad
        data = tensor_a._data ** tensor_b._data
        z = Tensor(data, requires_grad=requires_grad, operation=self)
        tensor_a.children.append(z)
        self.cache = (tensor_a, tensor_b)
        return z
    
    def backward(self, dz, z):
        tensor_a, tensor_b = self.cache
        if tensor_a.requires_grad:
            da = dz * (tensor_b._data * tensor_a._data ** (tensor_b._data-1))
            grad_dim = len(da.shape)
            in_dim = len(tensor_a.shape)
            for _ in range(grad_dim - in_dim):
                da = da.sum(axis=0)
            for n, dim in enumerate(tensor_a.shape):
                if dim == 1:
                    da = da.sum(axis=n, keepdims=True)
            tensor_a.backward(da, z)

class MatMul:

    def forward(self, a, b):
        requires_grad = a.requires_grad or b.requires_grad
     
        # Get new Tensor's data:
        data = a._data @ b._data
      
        # Create new Tensor:
        z = Tensor(data, requires_grad=requires_grad, operation=self) 
      
        # Add new Tensors to "children" and old Tensors to "parents":
        self.parents = (a, b)
        a.children.append(z)
        b.children.append(z)
        self.cache = (a, b)

        return z  
    
    def backward(self, dz, z):
        a, b = self.cache
        
        # Find gradients relative to "a", and pass it downstream:
        if a.requires_grad:
            # Backprop through the matmul:
            da = dz @ b._data.swapaxes(-1,-2)
            
            # Get difference between "a" size and upstream "da" size, to broadcast grad into "a":
            in_dim = len(a.shape)
            grad_dim = len(da.shape)

            for _ in range(grad_dim - in_dim):
                da = da.sum(axis=0)

            a.backward(da, z)

        # Find gradients relative to "b", and pass it downstream:
        if b.requires_grad:
            # Backprop through the matmul:
            db = a._data.swapaxes(-1,-2) @ dz

            # Get difference between "b" size and upstream "db" size, to broadcast grad into "b":
            in_dim = len(b.shape)
            grad_dim = len(db.shape)


            for _ in range(grad_dim - in_dim):
                db = db.sum(axis=0)

            b.backward(db, z)


# Element-wise operations:
class Exp:

    def forward(self, a):
        requires_grad = a.requires_grad
       
        # Get new Tensor's data:
        data = np.exp(a._data)
       
        # Create new Tensor:
        z = Tensor(data, requires_grad=requires_grad, operation=self) 
      
        # Add new Tensors to "children" and old Tensors to "parents":
        self.parents = (a,)
        a.children.append(z)
        self.cache = (a, data)

        return z
    
    def backward(self, dz, z):
        a, data = self.cache
        
        # Find gradients relative to "a", and pass it downstream:
        if a.requires_grad:
            # d/da(e^a) = e^a, apply the chain rule to the derivative of e^a:
            da = data * dz
            a.backward(da, z)


class Log:

    def forward(self, a):
        requires_grad = a.requires_grad
     
        # Get new Tensor's data:
        data = np.log(a._data)
     
        # Create new Tensor:
        z = Tensor(data, requires_grad=requires_grad, operation=self) 
      
        # Add new Tensors to "children" and old Tensors to "parents":
        self.parents = (a,)
        a.children.append(z)
        self.cache = (a)

        return z
    
    def backward(self, dz, z):
        a = self.cache
        
        # Find gradients relative to "a", and pass it downstream:
        if a.requires_grad:
            # d/da(ln(a)) = (1/a), apply the chain rule to the derivative of the natural log:
            da = (1 / a._data) * dz
            a.backward(da, z)


class Sqrt:

    def forward(self, a):
        requires_grad = a.requires_grad
     
        # Get new Tensor's data:
        data = np.sqrt(a._data)
     
        # Create new Tensor:
        z = Tensor(data, requires_grad=requires_grad, operation=self) 
     
        # Add new Tensors to "children" and old Tensors to "parents":
        self.parents = (a,)
        a.children.append(z)
        self.cache = (a, data)

        return z
    
    def backward(self, dz, z):
        a, data = self.cache
        
        # Find gradients relative to "a", and pass it downstream:
        if a.requires_grad:
            # d/dx(sqrt(a)) = (1/2) * (1/sqrt(a)), apply the chain rule to the derivative of the square root:
            da = (1 / 2) * (1 / data) * dz
            a.backward(da, z)

# Statistics operations:
class Sum:

    def forward(self, a, dim, keepdims):
        requires_grad = a.requires_grad
     
        # Get new Tensor's data:
        data = a._data.sum(axis=dim, keepdims=keepdims)
     
        # Create new Tensor:
        z = Tensor(data, requires_grad=requires_grad, operation=self) 
      
        # Add new Tensors to "children" and old Tensors to "parents":
        self.parents = (a,)
        a.children.append(z)
        self.cache = (a, dim, keepdims)

        return z
    
    def backward(self, dz, z):
        a, dim, keepdims =  self.cache
        
        # Find gradients relative to "a", and pass it downstream:
        if a.requires_grad:
            # If keepdims=False, the upstream gradient dz is missing the reduced dimension.
            # We must add it back to broadcast correctly.
            if not keepdims:
                dz = np.expand_dims(dz, axis=dim)
            
            # Now we can broadcast the gradient back to the original shape
            da = np.ones_like(a._data) * dz
            a.backward(da, z)


class Mean:

    def forward(self, a, dim, keepdims):
        requires_grad = a.requires_grad
    
        # Get new Tensor's data:
        data = a._data.mean(axis=dim, keepdims=keepdims)
      
        # Create new Tensor:
        z = Tensor(data, requires_grad=requires_grad, operation=self) 
       
        # Add new Tensors to "children" and old Tensors to "parents":
        self.parents = (a,)
        a.children.append(z)
        self.cache = (a, dim, keepdims)

        return z
    
    def backward(self, dz, z):
        a, dim, keepdims =  self.cache
        
        # Find gradients relative to "a", and pass it downstream:
        if a.requires_grad:
            if not keepdims:
                dz = np.expand_dims(dz, axis=dim)

            if dim is None:
                size = a._data.size
            else:
                size = a.shape[dim]
            
            da = (1 / size) * dz
            da = np.ones_like(a._data) * da
            a.backward(da, z)


class Max:

    def forward(self, a, dim, keepdims=False):
        requires_grad = a.requires_grad
      
        # Get new Tensor's data:
        data = np.max(a._data, axis=dim, keepdims=keepdims)
        
        # Create new Tensor:
        z = Tensor(data, requires_grad=requires_grad, operation=self) 
     
        # Add new Tensors to "children" and old Tensors to "parents":
        self.parents = (a,)
        a.children.append(z)
        self.cache = (a, dim, keepdims)

        return z
    
    def backward(self, dz, z):
        a, dim, keepdims =  self.cache

        # Find gradients relative to "a", and pass it downstream:
        if a.requires_grad:
            # Create a mask that is True only at the position of the first max value.
            # This handles ties by only backpropagating to the first occurrence.
            # CuPy does not have `put_along_axis`, so we replicate it by creating a one-hot mask.
            max_indices = np.argmax(a._data, axis=dim)
            
            # Create a one-hot encoding of the max indices.
            # This is equivalent to what np.put_along_axis(..., True) would do.
            one_hot_mask = np.eye(a.shape[dim])[max_indices]
            
            # Reshape the one-hot mask to match the original tensor's shape for broadcasting.
            mask = np.moveaxis(one_hot_mask, -1, dim).astype(bool)
            
            # If keepdims=False was used, the upstream gradient `dz` will be missing
            # the dimension that was reduced. We need to add it back for broadcasting.
            if not keepdims:
                dz = np.expand_dims(dz, axis=dim)

            # Route the broadcasted gradient only to the max values.
            da = mask * dz
            a.backward(da, z)
            

class Var:

    def forward(self, a, dim, keepdims):
        requires_grad = a.requires_grad
     
        # Get new Tensor's data:
        data = a._data.var(axis=dim, keepdims=keepdims)
      
        # Create new Tensor:
        z = Tensor(data, requires_grad=requires_grad, operation=self) 
      
        # Add new Tensors to "children" and old Tensors to "parents":
        self.parents = (a,)
        a.children.append(z)
        self.cache = (a, dim, keepdims)

        return z
    
    def backward(self, dz, z):
        a, dim, keepdims =  self.cache
        
        # Find gradients relative to "a", and pass it downstream:
        if a.requires_grad:
            if not keepdims:
                dz = np.expand_dims(dz, axis=dim)

            if dim is None:
                n = a._data.size
            else:
                n = a.shape[dim]
            
            grad_a = dz * (2 * (a._data - a._data.mean(axis=dim, keepdims=True))) / n
            a.backward(grad_a, z)


# Tensor Operations:
class Reshape:

    def forward(self, a, shape):
        requires_grad = a.requires_grad
      
        # Get new Tensor's data:
        data = a._data.reshape(*shape)
      
        # Create new Tensor:
        z = Tensor(data, requires_grad=requires_grad, operation=self)
      
        # Add new Tensors to "children" and old Tensors to "parents": 
        self.parents = (a,)
        a.children.append(z)
        self.cache = (a)

        return z
    
    def backward(self, dz, z):
        a = self.cache
        
        # Find gradients relative to "a", and pass it downstream:
        if a.requires_grad:
            # Reshape upstream gradients:
            da = dz.reshape(a.shape)
 
            a.backward(da, z)


class Transpose:

    def forward(self, a, *dims):
        requires_grad = a.requires_grad
       
        # Get new Tensor's data:
        data = a._data.swapaxes(*dims)
       
        # Create new Tensor:
        z = Tensor(data, requires_grad=requires_grad, operation=self)
       
        # Add new Tensors to "children" and old Tensors to "parents": 
        self.parents = (a,)
        a.children.append(z)
        self.cache = (a, dims)

        return z
    
    def backward(self, dz, z):
        a, dims = self.cache
        
        # Find gradients relative to "a", and pass it downstream:
        if a.requires_grad:
            # Transpose upstream gradients:
            da = dz.swapaxes(*dims)
 
            a.backward(da, z)


class Cat:

    def forward(self, tensors: tuple, dim: int):

        requires_grad = False
        for tensor in tensors:
            if tensor.requires_grad == True:
                requires_grad = True
    
        # Get new Tensor's data:
        data = np.concatenate([tensor._data for tensor in tensors], axis=dim)
    
        # Create new Tensor:
        z = Tensor(data, requires_grad=requires_grad, operation=self) 
    
        # Add new Tensors to "children" and old Tensors to "parents":
        self.parents = tensors
        for tensor in tensors:
            tensor.children.append(z)
        self.cache = (tensors, dim)

        return z
    
    def backward(self, dz, z):
        tensors, dim = self.cache
        
        dz = np.split(dz, len(tensors), dim)

        # Find gradients relative to each tensor in "tensor", and pass it downstream:
        for i, tensor in enumerate(tensors):
            if tensor.requires_grad:
                # For every tensor that generated the output, get gradients relative to that part of "dz": 
                di = dz[i]
    
                tensor.backward(di, z)


class Stack:
    
    def forward(self, tensors: tuple, dim: int):

        # Verify if any original tensors requires grad:
        requires_grad = False
        for tensor in tensors:
            if tensor.requires_grad == True:
                requires_grad = True
       
        # Get new Tensor's data:
        data = np.stack([tensor._data for tensor in tensors], axis=dim)
       
        # Create new Tensor:
        z = Tensor(data, requires_grad=requires_grad, operation=self) 
       
        # Add new Tensors to "children" and old Tensors to "parents":
        self.parents = tensors
        for tensor in tensors:
            tensor.children.append(z)
        self.cache = (tensors, dim)

        return z
    
    def backward(self, dz, z):
        tensors, dim = self.cache

        dz = np.split(dz, len(tensors), dim)

        # Find gradients relative to each tensor in "tensor", and pass it downstream:
        for i, tensor in enumerate(tensors):
            if tensor.requires_grad:
                # For every tensor that generated the stack, get gradients relative to that part of "dz": 
                di = dz[i].reshape(tensor._data.shape)
    
                tensor.backward(di, z)


class MaskedFill:

    def forward(self, a, condition, value):
        requires_grad = a.requires_grad
      
        # Get new Tensor's data: where condition is True, fill with value, otherwise keep original.
        data = np.where(condition, value, a._data)
      
        # Create new Tensor:
        z = Tensor(data, requires_grad=requires_grad, operation=self) 
      
        # Add new Tensors to "children" and old Tensors to "parents":
        self.parents = (a,)
        a.children.append(z)
        self.cache = (a, condition)

        return z 
    
    def backward(self, dz, z):
        a, condition = self.cache
        
        # Find gradients relative to "a", and pass it downstream:
        if a.requires_grad:
            # Gradients are 0 where the values were filled, and pass through otherwise.
            da = np.where(condition, 0, dz)
 
            a.backward(da, z)


class Slice:

    def forward(self, a, index):
        requires_grad = a.requires_grad
      
        # Get new Tensor's data:
        data = a._data[index]
       
        # Create new Tensor:
        z = Tensor(data, requires_grad=requires_grad, operation=self) 
       
        # Add new Tensors to "children" and old Tensors to "parents":
        self.parents = (a,)
        a.children.append(z)
        self.cache = (a, index)

        return z
    
    def backward(self, dz, z):
        a, index =  self.cache
        
        # Find gradients relative to "a", and pass it downstream:
        if a.requires_grad:
            # Add upstream gradients to [index] part of da.
            da = np.zeros_like(a._data)
            da[index] = dz
            a.backward(da, z)


# Some helper functions to transition between iterable data types:
def list(data):
    if isinstance(data, List):
        return data
    else: 
        return data.tolist()

def array(data):
    # If data is already a cupy array (when np is cupy), just return it
    if IS_GPU and isinstance(data, np.ndarray):
        return data
    if isinstance(data, Tensor):
        return data.toarray()
    
    # Use cupy's array function if on GPU, otherwise numpy's
    # This handles conversion from lists, numpy arrays, etc.
    if IS_GPU:
        # np.asarray avoids unnecessary copies if data is already a cupy array
        arr = np.asarray(data)
    else:
        arr = np.array(data)

    # Default to float32 for floating point numbers to save memory
    if np.issubdtype(arr.dtype, np.floating):
        return arr.astype(np.float32)
    return arr
    
def tensor(data):
    if isinstance(data, Tensor):
        return data
    else: 
        return Tensor(data)
