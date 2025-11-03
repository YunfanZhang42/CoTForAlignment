import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

X_LIM_DELTA = 5

# Helper
def load_scalar_series(log_path: str, tag: str):
    """Read a TensorBoard event file and return (x, y) series for plotting."""
    ea = event_accumulator.EventAccumulator(log_path,
                                            size_guidance={"scalars": 0})
    ea.Reload()

    scalar_events = ea.Scalars(tag)
    x = [i / len(scalar_events) * 100 for i in range(len(scalar_events))]
    y = [e.value for e in scalar_events]
    return x, y


# Per‑plot configuration
TAG = "val-core/steerable_vk/reward/mean@1"

plots = [
    {
        "log_path": "./tensorboard_llama3_vk/events.out.tfevents.1746119533.64-181-213-82.1412354.0",
        "ylim":     (0.55, 0.85),
        "refs": {                    # label -> y‑value
            "Llama 3 8B SFT":               0.759,
            "Llama 3 8B Human-written CoT": 0.781,
        },
        "x_label": "% of Training Data Seen on VK",
        "y_label": "Validation Set Accuracy",
    },
    {
        "log_path": "./tensorboard_llama3_opinionqa/events.out.tfevents.1746516053.64-181-213-82.1579812.0",
        "ylim":     (0.50, 0.75),
        "refs": {
            "Llama 3 8B SFT":        0.670,
            "Llama 3 8B Synthetic CoT": 0.627,
        },
        "x_label": "% of Training Data Seen on OpinionQA",
        "y_label": "",
    },
]


plt.style.use("seaborn-v0_8-talk")  # use the seaborn style

# Plotting
plt.rcParams.update({"font.size": 19, "font.weight": "bold"})        # make all text larger overall
fig, axes = plt.subplots(1, len(plots),
                         figsize=(6 * len(plots), 6),   # keep each panel 8×6
                         sharex=False, sharey=False)

# If only one panel, axes isn't a list
if len(plots) == 1:
    axes = [axes]

for ax, cfg in zip(axes, plots):
    # Main curve
    x, y = load_scalar_series(cfg["log_path"], TAG)
    ax.plot(x, y, linewidth=5, color="royalblue")

    # Axes formatting
    ax.set_xlabel(cfg["x_label"], fontsize=19, weight="bold")
    ax.set_xlim(0 - X_LIM_DELTA, 100 + X_LIM_DELTA)

    if cfg["y_label"]:
        ax.set_ylabel(cfg["y_label"], fontsize=19, weight="bold")
    ax.set_ylim(*cfg["ylim"])

    # Reference lines & labels
    offset = 0.001  # put text just above the line
    for label, y_val in cfg["refs"].items():
        ax.axhline(y_val, color="black", linestyle="--", linewidth=3)
        ax.text(100 + X_LIM_DELTA, y_val + offset, label,
                ha="right", va="bottom", fontsize=18, weight="bold")

    # Include the reference values in the y‑tick list
    ax.set_yticks(sorted(set(ax.get_yticks()).union(cfg["refs"].values())))

    # Grid
    # ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.7)

plt.tight_layout()
plt.savefig("rl_sample_efficiency.png", dpi=600,)
