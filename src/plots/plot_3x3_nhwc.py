import os

import matplotlib.pyplot as plt
import numpy as np


def plot_fusion_nhwc(sizes: np.ndarray, performance: np.ndarray):
    plt.style.use("ggplot")

    fig, ax = plt.subplots(figsize=[10, 7])

    ax.plot(sizes, performance[1], marker='s', markersize=12, linewidth=4, color="silver")
    plt.text(0, 0.06, "Fused", color="silver", weight="bold", fontsize=18, ha="left", va="center")

    ax.plot(sizes, performance[0], marker='o', markersize=12, linewidth=4, color="black")
    plt.text(800, 0.05, "Baseline", color="black", weight="bold", fontsize=18, ha="left", va="center")

    ax.plot(sizes, performance[2], marker='s', markersize=12, linewidth=4, color="gray")
    plt.text(70, 0.09, "SIMD packing: 3.4x", color="gray", weight="bold", fontsize=18, ha="left", va="center")

    # ax.plot(sizes, performance[3], marker='s', markersize=12, linewidth=6, color="darkgray")
    # plt.text(450, 0.083, "Blocked", color="darkred", weight="bold", fontsize=18, ha="left", va="center")

    ax.plot(sizes, performance[4], marker='s', markersize=12, linewidth=6, color="darkred")
    plt.text(600, 0.15, "Large Buffer: 2x", color="darkred", weight="bold", fontsize=18, ha="left", va="center")
    plt.text(80, 0.25, "max. speedup: 3.9x", color="darkred", weight="bold", fontsize=16, ha="left", va="center", style="italic")

    # plt.xticks(ticks=n, rotation=0)
    ax.set_xlabel("Input size $H = W$", fontsize=16, color="black")

    plt.title("Performance [$C_{quant}$ / cycle]", fontsize=16, color="black")
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

    plt.savefig("plots/nhwc_3x3.jpg", bbox_inches="tight")
    plt.show()
    return


def compute_intops_ternarize(N, C, H, W):
    # compute number of relevant ops for ternarization
    quantize = N * H * W * C

    return quantize


if __name__ == "__main__":
    # [----- Configuration Zone -----]
    PROCESSOR_FREQUENCY = 4.2  # in GHz
    MEASUREMENT_FACTOR = 1  # 1 (NANO), 1000 (MICRO), 1000000 (MILLI)

    sizes = [16, 28, 56, 112, 256, 386, 512, 750, 1024]  # replace with input sizes

    # compute intops
    intops_input = np.array([compute_intops_ternarize(1, 64, s, s) for s in sizes])
    intops_weights = np.array([compute_intops_ternarize(16, 64, 3, 3) for s in sizes])
    intops = intops_input + intops_weights

    # put measurements here
    baseline_quantization = np.array(
        [133009, 312917, 1128447, 4408598, 28714415, 54281963, 115821291, 253705599, 533564903])
    baseline_img2row = np.array([6703, 28823, 84268, 326077, 1843928, 4615283, 16225843, 38736005, 71539806])

    fused_1_quantization = np.array([48105, 48652, 49077, 49295, 51253, 51969, 59941, 58946, 59767])
    fused_1_img2row = np.array([117946, 361946, 1439819, 5755146, 30773505, 70046007, 150003236, 318067602, 593701218])

    fused_simd_quantization = np.array([44086, 45460, 46666, 47020, 49512, 50329, 60151, 59169, 60214])
    fused_simd_img2row = np.array([26722, 82063, 329352, 1339435, 7730331, 17506546, 44983864, 95282569, 176884603])

    fused_2_quantization = np.array([44741, 46067, 46837, 47535, 49952, 50904, 61724, 59322, 60557])
    fused_2_img2row = np.array([41754, 128099, 533132, 2206061, 11522482, 26041046, 63153775, 134654364, 248094781])

    fused_3_quantization = np.array([43664, 45479, 46075, 47025, 49647, 50662, 61347, 59759, 59823])
    fused_3_img2row = np.array([5780, 17407, 66980, 312975, 2966236, 6531265, 21671200, 46032788, 85517708])

    runtimes = np.stack([baseline_quantization + baseline_img2row, fused_1_quantization + fused_1_img2row,
                         fused_simd_quantization + fused_simd_img2row, fused_2_quantization + fused_2_img2row,
                         fused_3_quantization + fused_3_img2row])

    # choose index to report speedup
    baseline_idx = 2
    version_idx = 4
    input_idx = 3

    # choose plot function
    plot_func = plot_fusion_nhwc
    # [------------------------------]

    runtimes = runtimes.astype(float)
    runtimes *= PROCESSOR_FREQUENCY * MEASUREMENT_FACTOR  # runtimes in cycles
    performance = intops / runtimes

    # reports speedup
    print(f"speedup: {performance[version_idx][input_idx] / performance[baseline_idx][input_idx]:.2f}")

    plot_func(np.array(sizes), performance)
