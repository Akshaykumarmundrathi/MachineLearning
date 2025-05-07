import numpy as np
from neuron import neuron
from recep_field import rf
from spike_train import encode_stochastic
from weight_initialization import learned_weights

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# PARAMETERS
T = 20
Pth = 6
m = 25  # 5x5 input
n = 5   # 5 output neurons (one per class)

# Init neurons
layer2 = [neuron() for _ in range(n)]

# Load weights
weight_matrix = learned_weights()
synapse = np.zeros((n, m))
for i in range(n):
    synapse[i] = weight_matrix[i]

# Load your dataset
from testingsnnmodel import matrices, labels

all_preds = []
all_labels = []
all_spike_counts = []

for idx, (matrix, true_label) in enumerate(zip(matrices, labels)):
    for x in layer2:
        x.initial()
    pot = rf(matrix)
    train = encode_stochastic(pot, T)

    spike_count = np.zeros((n, 1))
    active_pot = np.zeros((n, 1))
    f_spike = 0

    for t in range(T):
        for j, x in enumerate(layer2):
            if x.t_rest < t:
                x.P += np.dot(synapse[j], train[:, t])
                if x.P > x.Prest:
                    x.P -= x.D
                active_pot[j] = x.P

        # Lateral inhibition
        if f_spike == 0:
            high_pot = max(active_pot)
            if high_pot > Pth:
                f_spike = 1
                winner = np.argmax(active_pot)
                for s in range(n):
                    if s != winner:
                        layer2[s].P = layer2[s].Pmin

        # Check for spikes
        for j, x in enumerate(layer2):
            s = x.check()
            if s == 1:
                spike_count[j] += 1
                x.t_rest = t + x.t_ref

    pred = int(np.argmax(spike_count))
    all_preds.append(pred)
    all_labels.append(int(true_label))
    all_spike_counts.append(spike_count.flatten())

    print("Matrix:\n", matrix)
    print("Prediction:", pred, "Actual:", true_label)
    print("Spike counts:", spike_count.flatten())
    print("-" * 30)

# ---- METRICS ----
success = 0
false_pos = 0
always_firing = 0
no_learning = 0
inverse_learning = 0

for i, (label, pred, spike_counts) in enumerate(zip(all_labels, all_preds, all_spike_counts)):
    if pred == label:
        success += 1
    elif np.count_nonzero(spike_counts == np.max(spike_counts)) > 1:
        always_firing += 1
    elif np.max(spike_counts) == 0:
        no_learning += 1
    else:
        false_pos += 1
        # Inverse learning: e.g., label 0-3 predicted as 4 or vice versa
        if (label in [0,1,2,3] and pred == 4) or (label == 4 and pred in [0,1,2,3]):
            inverse_learning += 1

total = len(all_labels)
print(f"Test samples: {total}")
print(f"Success rate: {success/total:.3f}")
print(f"False positives: {false_pos/total:.3f}")
print(f"Always-firing: {always_firing/total:.3f}")
print(f"No learning: {no_learning/total:.3f}")
print(f"Inverse learning: {inverse_learning/total:.3f}")

# ---- CONFUSION MATRIX VISUALIZATION ----
cm = confusion_matrix(all_labels, all_preds, labels=[0,1,2,3,4])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0,1,2,3,4])
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

# ---- VISUALIZE SOME SAMPLE MATRICES ----
def plot_matrix(matrix, title="Permutation Matrix"):
    plt.imshow(matrix, cmap="binary", vmin=0, vmax=1)
    plt.title(title)
    plt.colorbar()
    plt.xticks(range(5))
    plt.yticks(range(5))
    plt.tight_layout()
    plt.show()

for i in range(3):  # Plot first 3 samples
    plot_matrix(matrices[i], title=f"True: {labels[i]}, Pred: {all_preds[i]}")

