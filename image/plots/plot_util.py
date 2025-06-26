import pickle
from scipy.stats import beta
import numpy as np
import matplotlib.pyplot as plt


def normalize_values(results_dict):
    for key, val in results_dict.items():
        if key == "coverage":
            results_dict[key] = [v / 100 for v in val]
        if key == "target_partition_coverage":
            for p in val.keys():
                val[p] = [v / 100 for v in val[p]]
        if key == "class_coverage":
            for p in val.keys():
                val[p] = [v / 100 for v in val[p]]
        if key == "partition_coverage":
            for p in val.keys():
                val[p] = [v / 100 for v in val[p]]

def plot_coverage_mcp(results_path, target_partition, shift_type, calibration_size=5000, coverage=0.9):
        
    with open(f'{results_path}/baseline.pkl', 'rb') as handle:
        baseline_results = pickle.load(handle)

    with open(f'{results_path}/mondrian.pkl', 'rb') as handle:
        mondrian_results = pickle.load(handle)

    # from percentage to [0, 1] range for plotting
    normalize_values(baseline_results)
    normalize_values(mondrian_results)

    if target_partition == "label":
        # pos = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30]
        label2target = {
            0: "All",
            1: "Airplane",
            2: "Automobile",
            3: "Bird",
            4: "Cat",
            5: "Deer",
            6: "Dog",
            7: "Frog",
            8: "Horse",
            9: "Ship",
            10: "Truck",
        }
    elif target_partition == "color":
        # pos = [0, 3, 6, 9, 12, 15, 18, 21, 24]
        label2target = {
            0: "All",
            1: "(0, 0, 0)",
            2: "(1, 0, 0)",
            3: "(1, 1, 0)",
            4: "(1, 1, 1)",
            5: "(1, 0, 1)",
            6: "(0, 1, 0)",
            7: "(0, 1, 1)",
            8: "(0, 0, 1)",
        }
    elif target_partition == "entropy":
        # pos = [0, 3, 6, 9, 12, 15, 18]
        label2target = {
            0: "All",
            1: r"$1.0e-4$",
            2: r"$1.0e-3$",
            3: r"$1.0e-2$",
            4: r"$1.0e-1$",
            5: r"$1.0e-0$",
            6: r"$1.0e+1$",
        }

    pos = list(range(0, (len(label2target)-1) * 3 + 1, 3))
    space_between = 0.4
    data = [cov for cov in baseline_results["target_partition_coverage"].values()]

    
    data.insert(0, baseline_results["coverage"])

    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(10, 4))


    ql, qh = 0.01, 0.99 #(the targeted quantiles to draw the grey area)
    alpha = 0.1 #(or any other miscoverage level)
    n = calibration_size #(calibration set size)

    # compute cov beta distr
    l = np.floor((n+1)*alpha)
    a = n + 1 - l
    b = l
    rv = beta(a, b)

    # compute beta quantile
    q_low = rv.ppf(ql)
    q_high = rv.ppf(qh)

    shared_params = {
        "showfliers": False,
        "widths": 0.7,
        "notch": False,
        "whis": 0.5
    }

    pos_base = [p - space_between for p in pos]

    
    base_boxplot = axs.boxplot(
        data, 
        positions=pos_base, 
        showfliers=shared_params["showfliers"],
        notch=shared_params["notch"],
        patch_artist=True,
        whis=shared_params["whis"]
    )


    for patch in base_boxplot["boxes"]:
        patch.set_facecolor("red")
    for patch in base_boxplot["medians"]:
        patch.set_color("black")

    # data = [cov for cov in mondrian_results["class_coverage"].values()]
    data = [cov for cov in mondrian_results["target_partition_coverage"].values()]

    data.insert(0, mondrian_results["coverage"])

    pos_ours = [p + space_between for p in pos]
    ours_boxplot = axs.boxplot(
        data, 
        positions=pos_ours, 
        showfliers=shared_params["showfliers"],
        notch=shared_params["notch"],
        patch_artist=True,
        whis=shared_params["whis"]
    )

    for patch in ours_boxplot["boxes"]:
        patch.set_facecolor("cornflowerblue")

    for patch in ours_boxplot["medians"]:
        patch.set_color("black")

    axs.fill_between([pos[0] - 1.0, pos[-1] + 1.0], [q_low, q_low], y2=[q_high, q_high], color="silver", zorder=-5)
    axs.hlines(coverage, xmin=axs.get_xlim()[0], alpha=1.0, xmax=axs.get_xlim()[1], lw=2, ls="--", color="black", zorder=1)

    axs.set_xlim(pos[0] - 1.0 , pos[-1] + 1.0)

    # axs.set_yticks([0.75, 0.80, 0.85, 0.90, 0.95])
    # axs.set_ylim(0.8, 0.97)
    # min_y, max_y = axs.get_ylim()
    # axs.set_ylim()
    axs.set_xticks(pos)
    axs.set_xticklabels(list(label2target.values()), rotation=45)

    for tick in axs.xaxis.get_majorticklabels():
        tick.set_horizontalalignment("right")
                

    axs.set_ylabel("Coverage", fontdict={"size": 30})
    axs.legend([base_boxplot["boxes"][0], ours_boxplot["boxes"][0]], ['SCP', 'MCP'], loc="best")#loc='lower right')
    axs.grid(axis="y", lw=2, zorder=-4)
    axs.set_axisbelow(True)
    # change all spines
    for axis in ['top','bottom','left','right']:
        axs.spines[axis].set_linewidth(3)
    plt.show()
    plt.tight_layout()
