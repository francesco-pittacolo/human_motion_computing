import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GroupKFold

from load_dataset import MOVEMENT_NAMES, load_ksas
from utils_v2 import (
    FIGURES_DIR,
    MAIN_MAX_KERNEL_SPAN,
    N_SPLITS,
    RANDOM_STATE,
    RESULTS_DIR,
    build_unique_feature_index,
    encode_dataset,
    fit_encoder,
    make_encoder,
)


TOP_N = 20
PLOT_TOP_N = 5
KERNEL_LENGTH = 9

PATTERN_FIGURES_DIR = FIGURES_DIR / "task_1_3_patterns"
PATTERN_FIGURES_DIR.mkdir(exist_ok=True)

np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)


def pattern_positions(feature_index, pattern, dilation, channel):
    mask = (
        (feature_index.get_level_values("pattern") == pattern)
        & (feature_index.get_level_values("dilation") == dilation)
        & (feature_index.get_level_values("channel") == channel)
    )
    return np.flatnonzero(mask)


def plot_activation_by_class(values, labels, title, output_path):
    classes = sorted(np.unique(labels))
    grouped = [values[labels == class_id] for class_id in classes]
    names = [f"{class_id} - {MOVEMENT_NAMES[int(class_id)]}" for class_id in classes]

    plt.figure(figsize=(11, 5))
    plt.boxplot(grouped, tick_labels=names)
    plt.xlabel("Movement")
    plt.ylabel("Standardized XROCKET activation")
    plt.title(title)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def print_kernel_coverage():
    print("\nKernel coverage by dilation:")

    for dilation in range(1, 5):
        coverage = 1 + (KERNEL_LENGTH - 1) * dilation
        print(f"  dilation {dilation} -> {coverage} samples")


def print_fold_top_patterns(pattern_importance, top_n=5):
    top = pattern_importance.sort_values(ascending=False).head(top_n)

    print(f"Top {top_n} patterns in this fold:")

    for rank, (feature_key, importance) in enumerate(top.items(), start=1):
        _, dilation, channel = feature_key
        print(f"  {rank}. {channel} | dilation {int(dilation)} | importance {importance:.6f}")


def print_pattern_summary(pattern_summary):
    print("\n" + "=" * 70)
    print("TASK 1.3 - TOP DISCRIMINATIVE PATTERNS")
    print("=" * 70)

    for _, row in pattern_summary.iterrows():
        print()
        print(f"Pattern #{int(row['rank'])}")
        print(f"  Sensor channel          : {row['channel']}")
        print(f"  Dilation                : {int(row['dilation'])}")
        print(f"  Importance              : {row['mean_importance']:.6f} ± {row['std_importance']:.6f}")
        print(f"  Pattern                 : {row['pattern']}")
        print()
        print(f"  Highest mean activation : class {int(row['highest_mean_class'])} - {row['highest_mean_movement']}")
        print(f"  Lowest mean activation  : class {int(row['lowest_mean_class'])} - {row['lowest_mean_movement']}")
        print(f"  Normalized separation   : {row['normalized_separation']:.3f}")


def main():
    X, y, participants, _, _ = load_ksas()
    gkf = GroupKFold(n_splits=N_SPLITS)

    fold_importances = []
    fold_accuracies = []
    activation_data = []

    print("=" * 70)
    print("TASK 1.3 - DISCRIMINATIVE PATTERN ANALYSIS")
    print("=" * 70)
    print(f"Recordings: {len(X)}")
    print(f"Participants: {len(np.unique(participants))}")
    print(f"max_kernel_span: {MAIN_MAX_KERNEL_SPAN}")
    print("combination_order: 1")

    print_kernel_coverage()

    for fold, (train_idx, test_idx) in enumerate(gkf.split(np.zeros(len(y)), y, participants), start=1):
        X_train = [X[index] for index in train_idx]
        X_test = [X[index] for index in test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        print("\n" + "=" * 70)
        print(f"FOLD {fold}")
        print("=" * 70)

        encoder = make_encoder(MAIN_MAX_KERNEL_SPAN)
        fit_encoder(encoder, X_train[0])

        Z_train = encode_dataset(encoder, X_train)
        Z_test = encode_dataset(encoder, X_test)

        feature_index = build_unique_feature_index(encoder)

        if len(feature_index) != Z_train.shape[1]:
            raise RuntimeError("XROCKET feature descriptions do not match embedding dimensionality")

        rf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
        rf.fit(Z_train, y_train)

        predictions = rf.predict(Z_test)
        accuracy = accuracy_score(y_test, predictions)
        fold_accuracies.append(accuracy)

        print(f"RF test accuracy: {accuracy:.3f}")

        raw_importance = pd.Series(rf.feature_importances_, index=feature_index)
        pattern_importance = raw_importance.groupby(level=["pattern", "dilation", "channel"]).sum()
        pattern_importance.name = f"fold_{fold}"
        fold_importances.append(pattern_importance)

        train_mean = Z_train.mean(axis=0)
        train_std = Z_train.std(axis=0)
        train_std[train_std < 1e-8] = 1.0
        Z_test_standardized = (Z_test - train_mean) / train_std

        activation_data.append({
            "embeddings": Z_test_standardized,
            "labels": y_test,
            "feature_index": feature_index,
        })

        print_fold_top_patterns(pattern_importance)

    importance = pd.concat(fold_importances, axis=1).fillna(0.0)
    fold_columns = [f"fold_{fold}" for fold in range(1, N_SPLITS + 1)]

    importance["mean_importance"] = importance[fold_columns].mean(axis=1)
    importance["std_importance"] = importance[fold_columns].std(axis=1)
    importance = importance.sort_values("mean_importance", ascending=False)

    top_patterns = importance.head(TOP_N).copy()

    profile_rows = []
    summary_rows = []

    for rank, feature_key in enumerate(top_patterns.index, start=1):
        pattern, dilation, channel = feature_key
        values_by_fold = []
        labels_by_fold = []

        for fold_data in activation_data:
            positions = pattern_positions(fold_data["feature_index"], pattern, dilation, channel)

            if len(positions) == 0:
                continue

            values_by_fold.append(fold_data["embeddings"][:, positions].mean(axis=1))
            labels_by_fold.append(fold_data["labels"])

        if not values_by_fold:
            continue

        values = np.concatenate(values_by_fold)
        labels = np.concatenate(labels_by_fold)
        class_means = {}

        for class_id in sorted(np.unique(labels)):
            class_values = values[labels == class_id]
            class_id = int(class_id)
            class_means[class_id] = class_values.mean()

            profile_rows.append({
                "rank": rank,
                "pattern": pattern,
                "dilation": int(dilation),
                "channel": channel,
                "class": class_id,
                "movement": MOVEMENT_NAMES[class_id],
                "count": len(class_values),
                "mean_activation": class_values.mean(),
                "std_activation": class_values.std(ddof=1),
                "median_activation": np.median(class_values),
            })

        highest_class = max(class_means, key=class_means.get)
        lowest_class = min(class_means, key=class_means.get)

        activation_range = class_means[highest_class] - class_means[lowest_class]
        global_std = values.std(ddof=1)
        separation = activation_range / global_std if global_std > 0 else 0.0

        summary_rows.append({
            "rank": rank,
            "pattern": pattern,
            "dilation": int(dilation),
            "channel": channel,
            "mean_importance": top_patterns.loc[feature_key, "mean_importance"],
            "std_importance": top_patterns.loc[feature_key, "std_importance"],
            "highest_mean_class": highest_class,
            "highest_mean_movement": MOVEMENT_NAMES[highest_class],
            "lowest_mean_class": lowest_class,
            "lowest_mean_movement": MOVEMENT_NAMES[lowest_class],
            "activation_range": activation_range,
            "normalized_separation": separation,
        })

    class_profiles = pd.DataFrame(profile_rows)
    pattern_summary = pd.DataFrame(summary_rows)

    importance.to_csv(RESULTS_DIR / "task_1_3_pattern_importance_across_folds.csv")
    class_profiles.to_csv(RESULTS_DIR / "task_1_3_class_profiles.csv", index=False)
    pattern_summary.to_csv(RESULTS_DIR / "task_1_3_pattern_summary.csv", index=False)

    print("\n" + "=" * 70)
    print("CROSS-VALIDATION ACCURACY")
    print("=" * 70)

    for fold, accuracy in enumerate(fold_accuracies, start=1):
        print(f"Fold {fold}: {accuracy:.3f}")

    print(f"Mean: {np.mean(fold_accuracies):.3f}")
    print(f"Std. dev.: {np.std(fold_accuracies, ddof=1):.3f}")

    print_pattern_summary(pattern_summary)

    plot_data = pattern_summary.head(10).copy()
    plot_data["label"] = [f"#{int(row['rank'])} {row['channel']}\nd={int(row['dilation'])}" for _, row in plot_data.iterrows()]
    plot_data = plot_data.sort_values("mean_importance")

    plt.figure(figsize=(10, 7))
    plt.barh(plot_data["label"], plot_data["mean_importance"], xerr=plot_data["std_importance"], capsize=3)
    plt.xlabel("Mean Random Forest feature importance")
    plt.ylabel("XROCKET pattern")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "task_1_3_top_pattern_importance.png", dpi=300, bbox_inches="tight")
    plt.close()

    for _, row in pattern_summary.head(PLOT_TOP_N).iterrows():
        rank = int(row["rank"])
        pattern = row["pattern"]
        dilation = int(row["dilation"])
        channel = row["channel"]

        all_values = []
        all_labels = []

        for fold_data in activation_data:
            positions = pattern_positions(fold_data["feature_index"], pattern, dilation, channel)

            if len(positions) == 0:
                continue

            all_values.append(fold_data["embeddings"][:, positions].mean(axis=1))
            all_labels.append(fold_data["labels"])

        if not all_values:
            continue

        values = np.concatenate(all_values)
        labels = np.concatenate(all_labels)
        title = f"Pattern #{rank} - {channel}, dilation {dilation}"

        plot_activation_by_class(
            values,
            labels,
            title,
            PATTERN_FIGURES_DIR / f"pattern_{rank:02d}_by_class.png",
        )

    print("\n" + "=" * 70)
    print("FILES CREATED")
    print("=" * 70)
    print("results/task_1_3_pattern_importance_across_folds.csv")
    print("results/task_1_3_class_profiles.csv")
    print("results/task_1_3_pattern_summary.csv")
    print("figures/task_1_3_top_pattern_importance.png")
    print("figures/task_1_3_patterns/pattern_01_by_class.png ... pattern_05_by_class.png")


if __name__ == "__main__":
    main()