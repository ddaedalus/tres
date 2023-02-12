import pickle
import matplotlib.pyplot as plt
import numpy as np

with open("times.pickle", "rb") as fp:
    times = pickle.load(fp)


plt.plot(np.arange(len(times)), np.cumsum(times) / 3600, "r")
plt.xlabel("Timesteps")
plt.ylabel("Hours")
plt.savefig("times.png", dpi=150)
