import numpy as np

def rf(inp):
    # For 5x5 permutation matrices, just return as float
    assert inp.shape == (5,5), f"Input matrix must be 5x5, got {inp.shape}"
    return inp.astype(np.float32)
