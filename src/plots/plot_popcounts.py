import os
import matplotlib.pyplot as plt
import numpy as np

def plot_popcounts_per_cycle(sizes: np.ndarray, popcounts_baseline: np.ndarray, popcounts_simd: np.ndarray, popcounts_popcount_simd: np.ndarray, popcounts_opt: np.ndarray):
    plt.style.use("ggplot")

    fig, ax = plt.subplots(figsize=[10, 7])

    ax.plot(sizes, popcounts_baseline, marker='o', markersize=12, linewidth=4, color="black", label="Baseline")
    ax.plot(sizes, popcounts_simd, marker='s', markersize=12, linewidth=4, color="gray", label="SIMD")
    ax.plot(sizes, popcounts_popcount_simd, marker='d', markersize=12, linewidth=4, color="darkgray", label="Popcount SIMD")
    ax.plot(sizes, popcounts_opt, marker='x', markersize=12, linewidth=4, color="darkred", label="Opt")


    ax.set_xlabel("N = K", fontsize=16, color="black")
    ax.set_ylabel("Popcounts per cycle", fontsize=16, color="black")

    plt.title("Popcounts per Cycle", fontsize=16, color="black")
    ax.set_ylim(0.1, 0.6)
    ax.set_xscale('log')

    ax.spines["bottom"].set_color("black")
    ax.spines["bottom"].set_linewidth(2)
    for tick in ax.get_yticks():
        ax.axhline(y=tick, color="white", zorder=0)

    ax.grid(axis='x', linewidth=0)
    ax.grid(axis='y', linewidth=2)

    ax.tick_params(axis="both", width=2, color="black")

    ax.set_xticks([100, 500, 1000, 5000, 10000, 20000])
    ax.set_xticklabels(['100', '500', '1K', '5K', '10K', '20K'])

    ax.legend(loc='lower right', fontsize=16)

    directory = "plots"

    if not os.path.exists(directory):
        os.makedirs(directory)

    plt.savefig("plots/popcounts_per_cycle.jpg", bbox_inches="tight")
    plt.show()
    return

if __name__ == "__main__":
    sizes = np.array([100, 500, 1000, 5000, 10000, 20000])
    popcounts_baseline = np.array([0.57, 0.57, 0.47, 0.44, 0.42, 0.41])
    popcounts_simd = np.array([0.52, 0.52, 0.50, 0.37, 0.36, 0.31])
    popcounts_popcount_simd = np.array([0.14, 0.39, 0.47, 0.42, 0.37, 0.33])
    popcounts_opt = np.array([0.57, 0.57, 0.56, 0.55, 0.53, 0.45])

    plot_popcounts_per_cycle(sizes, popcounts_baseline, popcounts_simd, popcounts_popcount_simd, popcounts_opt)
