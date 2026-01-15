import numpy as np
import matplotlib.pyplot as plt

data = np.load("data/raw/halo/0.npy")

plt.plot(data[:, 0], label="x landmark 0")
plt.plot(data[:, 1], label="y landmark 0")
plt.legend()
plt.title("Motion check")
plt.show()