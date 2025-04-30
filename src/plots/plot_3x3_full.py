import os

import matplotlib.pyplot as plt
import numpy as np


def plot_3x3(sizes: np.ndarray, performance: np.ndarray):
    plt.style.use("ggplot")

    fig, ax = plt.subplots(figsize=[10, 7])

    ax.plot(sizes, performance[0], marker='o', markersize=12, linewidth=4, color="black")
    plt.text(450, 0.57, "Baseline", color="black", weight="bold", fontsize=18, ha="left", va="center")

    # ax.plot(sizes, performance[1], marker='s', markersize=12, linewidth=4, color="silver")
    # plt.text(0, 0.06, "Fused", color="silver", weight="bold", fontsize=18, ha="left", va="center")

    # ax.plot(sizes, performance[2], marker='s', markersize=12, linewidth=4, color="gray")
    # plt.text(70, 0.09, "SIMD packing: 7.8x", color="gray", weight="bold", fontsize=18, ha="left", va="center")

    ax.plot(sizes, performance[3], marker='s', markersize=12, linewidth=4, color="darkred")
    plt.text(200, 0.78, "Our version: 1.3x", color="darkred", weight="bold", fontsize=18, ha="left", va="center")

    # plt.xticks(ticks=n, rotation=0)
    ax.set_xlabel("Input size $H = W$", fontsize=16, color="black")

    plt.title("Performance [$C_{total}$ / cycle]", fontsize=16, color="black")
    ax.set_ylim(0, None)

    # ax.set_title("Performance Measurements", fontsize=22, color="black", pad=30)

    ax.spines["bottom"].set_color("black")
    ax.spines["bottom"].set_linewidth(2)
    for tick in ax.get_yticks():
        ax.axhline(y=tick, color="white", zorder=0)

    ax.grid(axis='x', linewidth=0)
    ax.grid(axis='y', linewidth=2)

    ax.tick_params(axis="both", width=2, color="black")

    directory = "plots"

    if not os.path.exists(directory):
        os.makedirs(directory)

    plt.savefig("plots/full_3x3.jpg", bbox_inches="tight")
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
    MEASUREMENT_FACTOR = 1  # 1 (NANO), 1000 (MICRO), 1000000 (MILLI)

    sizes = [16, 28, 56, 112, 256, 386, 512, 600]  # replace with input sizes

    # compute intops for 3 layers
    intops = 3 * np.array([compute_intops(1, 64, s, s, 16, 3, 3, 1, 1) for s in sizes])

    # put measurements here
    baseline = np.array(
        [471567, 1034116, 3550512, 13637659, 76736070, 163793206, 315551063, 407181073])

    fused_1 = np.array([492251, 1108853, 3822621, 14726982, 82007678, 178109624, 334852599, 437126322])

    fused_2 = np.array([238500, 680994, 2684024, 11005077, 60881954, 132584560, 252258386, 334945909])

    fused_3 = np.array([234238, 668566, 2625170, 10731544, 59735833, 129834658, 247626382, 328518781])

    runtimes = np.stack([baseline, fused_1, fused_2, fused_3])

    # choose index to report speedup
    baseline_idx = 0
    version_idx = 3
    input_idx = 3

    # choose plot function
    plot_func = plot_3x3
    # [------------------------------]

    runtimes = runtimes.astype(float)
    runtimes *= PROCESSOR_FREQUENCY * MEASUREMENT_FACTOR  # runtimes in cycles
    performance = intops / runtimes

    # reports speedup
    print(f"speedup: {performance[version_idx][input_idx] / performance[baseline_idx][input_idx]:.2f}")

    plot_func(np.array(sizes), performance)
