import numpy as np
import matplotlib.pyplot as plt

def permutation_matrix_from_order(order):
    """Create a 5x5 permutation matrix from a list of indices (0-based)."""
    mat = np.zeros((5, 5), dtype=int)
    for col, row in enumerate(order):
        mat[row, col] = 1
    return mat

# Define the canonical orders for each class
orders = [
    [0, 1, 2, 3, 4],  # Strictly Increasing
    [4, 3, 2, 1, 0],  # Strictly Decreasing
    [4, 2, 0, 2, 4],  # V-shaped
    [0, 2, 4, 2, 0],  # Inverted V-shaped
    [1, 3, 0, 4, 2],  # Mixed/Random
]

class_names = [
    "Strictly Increasing",
    "Strictly Decreasing",
    "V-shaped",
    "Inverted V-shaped",
    "Mixed/Random"
]

# Generate and plot all 5 classes
fig, axes = plt.subplots(1, 5, figsize=(15, 3))
for i, (order, name) in enumerate(zip(orders, class_names)):
    mat = permutation_matrix_from_order(order)
    axes[i].imshow(mat, cmap='gray_r', vmin=0, vmax=1)
    axes[i].set_title(name)
    axes[i].set_xticks([])
    axes[i].set_yticks([])
    for (x, y), value in np.ndenumerate(mat):
        if value:
            axes[i].text(y, x, '1', ha='center', va='center', color='red')
plt.tight_layout()
plt.show()

# If you want the matrices as numpy arrays for your SNN:
matrices = [permutation_matrix_from_order(order) for order in orders]
