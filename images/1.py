# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-4, 4, 100)
# y = [(1 - np.exp(-2t)) / (1 + np.exp(2t)) for t in x]
y = (1 - np.exp(-2 * x)) / (1 + np.exp(-2 * x))
print(len(y))

plt.plot(x, y, linewidth=2)
plt.hlines(0, -4, 4)
# plt.axhline(0, c='black')
plt.axvline(0, c='black')
plt.xlim(-4, 4)
plt.ylim(-1, 1)
plt.grid()
plt.title("Tanh")
plt.savefig("tanh.png", dpi=300, format='png')
plt.show()
