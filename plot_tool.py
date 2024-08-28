import numpy as np
import matplotlib.pyplot as plt


def plot_result(all_pops, options):
    all_pops = np.array(all_pops)
    # x = np.arange(0, options["max_gen"])
    # y = np.array(all_pops_results)
    x = all_pops[:, 0]
    y = all_pops[:, 1]
    # y = np.array(all_pops)
    plt.title("...")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.plot(x, y, color="red",)
    plt.show()