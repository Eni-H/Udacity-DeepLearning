"""Softmax."""

import numpy as np

scores = np.array([[1, 2, 3, 6],
                   [2, 4, 5, 6],
                   [3, 8, 7, 6]])

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x)/np.sum(np.exp(x),axis=0)

print('Softmax matrix for scores: {}'.format(softmax(scores)))
print('Softmax matrix for scores*10: {}'.format(softmax(scores*10)))
print('Softmax matrix for scores/10: {}'.format(softmax(scores/10)))

# Plot softmax curves
import matplotlib.pyplot as plt
x = np.arange(-2.0, 6.0, 0.1)
scores = np.vstack([x, np.ones_like(x), 0.2 * np.ones_like(x)])

plt.plot(x, softmax(scores).T, linewidth=2)
plt.show()