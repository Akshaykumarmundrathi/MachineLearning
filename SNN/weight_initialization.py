import numpy as np

def learned_weights():
    # Canonical patterns for each class
    patterns = [
        # Label 0: Strictly increasing (bottom-left to top-right)
        np.array([
            [0,0,0,0,1],
            [0,0,0,1,0],
            [0,0,1,0,0],
            [0,1,0,0,0],
            [1,0,0,0,0]
        ]),
        # Label 1: Strictly decreasing (top-left to bottom-right)
        np.array([
            [1,0,0,0,0],
            [0,1,0,0,0],
            [0,0,1,0,0],
            [0,0,0,1,0],
            [0,0,0,0,1]
        ]),
        # Label 2: ^-shaped (increase then decrease)
        np.array([
            [0,0,1,0,0],
            [0,0,1,0,0],
            [0,1,0,1,0],
            [0,1,0,1,0],
            [1,0,0,0,1]
        ]),
        # Label 3: V-shaped (decrease then increase)
        np.array([
            [1,0,0,0,1],
            [1,0,0,0,1],
            [0,1,0,1,0],
            [0,1,0,1,0],
            [0,0,1,0,0]
        ]),
        # Label 4: Mixed patterns (W/M/N/reverse N)
        np.array([
            [1,0,1,0,1],
            [0,1,0,1,0],
            [0,0,0,0,0],
            [0,1,0,1,0],
            [1,0,1,0,1]
        ])
    ]
    ans = []
    for pattern in patterns:
        temp = []
        for i in pattern:
            for j in i:
                temp.append(j)
        ans.append(temp)
    return ans
