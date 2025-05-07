import numpy as np

def encode_stochastic(pot, T=20):
    """
    Build probabilistic spike train for each pixel.
    pot: 5x5 matrix (float), values 0 or 1
    Returns: (25, T) spike train
    """
    flat = pot.flatten()
    train = []
    for val in flat:
        spikes = (np.random.rand(T) < val).astype(int)
        train.append(spikes)
    return np.array(train)
