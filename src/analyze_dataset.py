import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from load_dataset import MOVEMENT_NAMES, PROJECT_DIR, load_ksas


RESULTS_DIR = PROJECT_DIR / "results"
FIGURES_DIR = PROJECT_DIR / "figures"
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)


def main():
    X, y, participants, arms, _ = load_ksas()

    lengths = np.asarray([recording.shape[1] for recording in X])
    missing_values = int(sum(np.isnan(recording).sum() for recording in X))

    dataset_summary = pd.DataFrame({
        "n_recordings": [len(X)],
        "n_participants": [len(np.unique(participants))],
        "n_classes": [len(np.unique(y))],
        "n_channels": [X[0].shape[0]],
        "min_length": [int(lengths.min())],
        "max_length": [int(lengths.max())],
        "mean_length": [float(lengths.mean())],
        "median_length": [float(np.median(lengths))],
        "missing_values": [missing_values],
    })

    rows = []
    for class_id in sorted(np.unique(y)):
        class_lengths = lengths[y == class_id]
        rows.append({
            "class": int(class_id),
            "movement": MOVEMENT_NAMES[int(class_id)],
            "count": int(len(class_lengths)),
            "mean_length": float(class_lengths.mean()),
            "std_length": float(class_lengths.std(ddof=1)),
            "min_length": int(class_lengths.min()),
            "median_length": float(np.median(class_lengths)),
            "max_length": int(class_lengths.max()),
        })

    class_summary = pd.DataFrame(rows)

    arm_values, arm_counts = np.unique(arms, return_counts=True)
    arm_summary = pd.DataFrame({
        "arm": arm_values,
        "count": arm_counts,
    })

    dataset_summary.to_csv(RESULTS_DIR / "dataset_summary.csv", index=False)
    class_summary.to_csv(RESULTS_DIR / "dataset_class_summary.csv", index=False)

    print("=" * 70)
    print("DATASET ANALYSIS")
    print("=" * 70)
    print(dataset_summary.to_string(index=False))

    print("\n" + "=" * 70)
    print("SEQUENCE LENGTH BY CLASS")
    print("=" * 70)
    print(class_summary.to_string(index=False))

    print("\n" + "=" * 70)
    print("ARM COUNTS")
    print("=" * 70)
    print(arm_summary.to_string(index=False))

    plot_data = []
    plot_labels = []

    for class_id in sorted(np.unique(y)):
        plot_data.append(lengths[y == class_id])
        plot_labels.append(f"{class_id} - {MOVEMENT_NAMES[int(class_id)]}")

    plt.figure(figsize=(11, 5))
    plt.boxplot(plot_data, tick_labels=plot_labels)
    plt.xlabel("Movement class")
    plt.ylabel("Original sequence length (samples)")
    plt.title("Sequence length distribution by movement class")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "dataset_sequence_length_by_class.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("\nFiles created:")
    print("results/dataset_summary.csv")
    print("results/dataset_class_summary.csv")
    print("figures/dataset_sequence_length_by_class.png")


if __name__ == "__main__":
    main()