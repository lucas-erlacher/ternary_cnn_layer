import os

import matplotlib.pyplot as plt
import numpy as np


def plot_1x1(sizes: np.ndarray, performance: np.ndarray):
    plt.style.use("ggplot")

    fig, ax = plt.subplots(figsize=[10, 7])

    ax.plot(sizes, performance[0], marker='o', markersize=12, linewidth=4, color="black")
    plt.text(1000, 0.9, "Baseline", color="black", weight="bold", fontsize=18, ha="left", va="center")

    ax.plot(sizes, performance[1], marker='s', markersize=12, linewidth=4, color="gray")
    plt.text(150, 4.2, "Specialized v1", color="gray", weight="bold", fontsize=18, ha="left", va="center")

    ax.plot(sizes, performance[2], marker='s', markersize=12, linewidth=4, color="darkgray")
    plt.text(7000, 3.8, "Specialized v2", color="darkgray", weight="bold", fontsize=18, ha="left", va="center")

    plt.text(3000, 2.5, "max. speedup: 10x", color="darkgray", weight="bold", fontsize=18, ha="left", va="center", style='italic')

    ax.set_xlabel("Input size $C = KN$", fontsize=16, color="black")

    plt.title("Performance [$C_{total}$ / cycle]", fontsize=16, color="black")
    ax.set_ylim(0, None)
    ax.set_xscale('log')

    ax.spines["bottom"].set_color("black")
    ax.spines["bottom"].set_linewidth(2)
    for tick in ax.get_yticks():
        ax.axhline(y=tick, color="white", zorder=0)

    ax.grid(axis='x', linewidth=0)
    ax.grid(axis='y', linewidth=2)

    ax.tick_params(axis="both", width=2, color="black")

    ax.set_xticks([100, 500, 1000, 5000, 10000, 50000])
    ax.set_xticklabels(['100', '500', '1K', '5K', '10K', '50K'])

    directory = "plots"

    if not os.path.exists(directory):
        os.makedirs(directory)

    plt.savefig("plots/full_1x1.jpg", bbox_inches="tight")
    plt.show()
    return


def compute_intops(BN, C, H, W, KN, KH, KW, padding, stride):
    # compute number of relevant ops for ternarization
    priChannel = C // 64
    packC = priChannel
    if C % 64 != 0:
        packC += 1

    # quantization
    quantize_w = KN * KH * KW * C  # Assuming even distribution of {-1, 0, 1}
    quantize_x = BN * H * W * C

    padded_h = H + 2 * padding
    padded_w = W + 2 * padding

    OH = (padded_h - KH) // stride + 1
    OW = (padded_w - KW) // stride + 1

    M = BN * OH * OW
    N = KN
    K = packC * KH * KW

    # GEMM & PReLU
    gemm = M * N * K * (5 + 2)  # 5 intops + 2 popcnt
    gemm_acc = M * N * 2.5      # 2 for gemm + 0.5 for PReLU (assuming even distribution of values)

    return quantize_w + quantize_x + gemm + gemm_acc


if __name__ == "__main__":
    # [----- Configuration Zone -----]
    PROCESSOR_FREQUENCY = 4.2  # in GHz
    MEASUREMENT_FACTOR = 1  # 1 (NANO), 1_000 (MICRO), 1_000_000 (MILLI)

    sizes = [200, 500, 1000, 2000, 3000, 5000, 10000, 20000, 30000]  # replace with input sizes

    # compute intops for 3 layers
    intops = 3 * np.array([compute_intops(32, s, 1, 1, s, 1, 1, 0, 1) for s in sizes])

    # put measurements here
    baseline = np.array(
        [315531, 1736961, 6760058, 25243947, 55972831, 156882724, 640542965, 2559372446, 5755237534])

    specialized_v1 = np.array([81404, 260589, 814573, 3323350, 6494693, 18698229, 69147824, 270356358, 611586618])

    specialized_v2 = np.array([106205, 372193, 1008749, 4031501, 8551925, 22755992, 67241930, 254697120, 571594178])

    runtimes = np.stack([baseline, specialized_v1, specialized_v2])

    # choose index to report speedup
    baseline_idx = 0
    version_idx = 2
    input_idx = 8

    # choose plot function
    plot_func = plot_1x1
    # [------------------------------]

    runtimes = runtimes.astype(float)
    runtimes *= PROCESSOR_FREQUENCY * MEASUREMENT_FACTOR  # runtimes in cycles
    performance = intops / runtimes

    # reports speedup
    print(f"speedup: {performance[version_idx][input_idx] / performance[baseline_idx][input_idx]:.2f}")

    plot_func(np.array(sizes), performance)
