import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from load_dataset import CHANNEL_NAMES, DATASET_DIR, load_ksas
from utils import (
    FIGURES_DIR,
    MAIN_MAX_KERNEL_SPAN,
    N_SPLITS,
    RANDOM_STATE,
    RESULTS_DIR,
    encode_dataset,
    feature_metadata,
    fit_encoder,
    make_encoder,
)


np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)


def main():
    X, y, participants, _, _ = load_ksas()
    lengths = np.asarray([recording.shape[1] for recording in X])

    print("=" * 70)
    print("MAIN XROCKET ANALYSIS")
    print("=" * 70)
    print(f"Dataset directory    : {DATASET_DIR}")
    print("Input representation : original variable-length recordings")
    print(f"Max kernel span      : {MAIN_MAX_KERNEL_SPAN}")
    print("Combination order    : 1")
    print(f"Recordings           : {len(X)}")
    print(f"Participants         : {len(np.unique(participants))}")
    print(f"Length range         : {lengths.min()}-{lengths.max()} samples")
    print(f"Mean length          : {lengths.mean():.3f} samples")

    gkf = GroupKFold(n_splits=N_SPLITS)

    fold_rows = []
    channel_rows = []
    dilation_rows = []

    for fold, (train_idx, test_idx) in enumerate(gkf.split(np.zeros(len(y)), y, participants), start=1):
        X_train = [X[index] for index in train_idx]
        X_test = [X[index] for index in test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        train_participants = np.unique(participants[train_idx])
        test_participants = np.unique(participants[test_idx])

        if np.intersect1d(train_participants, test_participants).size:
            raise RuntimeError("Participant leakage detected")

        print("\n" + "=" * 70)
        print(f"FOLD {fold}")
        print("=" * 70)
        print("Train participants:", train_participants)
        print("Test participants :", test_participants)

        encoder = make_encoder(MAIN_MAX_KERNEL_SPAN)
        fit_encoder(encoder, X_train[0])

        Z_train = encode_dataset(encoder, X_train)
        Z_test = encode_dataset(encoder, X_test)
        metadata = feature_metadata(encoder)

        if len(metadata) != Z_train.shape[1]:
            raise RuntimeError("XROCKET feature metadata does not match embedding dimensionality")

        dilations = sorted(metadata["dilation"].unique())

        print("Dilations        :", dilations)
        print("Train embeddings :", Z_train.shape)
        print("Test embeddings  :", Z_test.shape)

        rf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
        rf.fit(Z_train, y_train)
        rf_test_pred = rf.predict(Z_test)

        scaler = StandardScaler()
        Z_train_scaled = scaler.fit_transform(Z_train)
        Z_test_scaled = scaler.transform(Z_test)

        lr = LogisticRegression(max_iter=5000, random_state=RANDOM_STATE)
        lr.fit(Z_train_scaled, y_train)
        lr_test_pred = lr.predict(Z_test_scaled)

        rf_acc = accuracy_score(y_test, rf_test_pred)
        rf_f1 = f1_score(y_test, rf_test_pred, average="macro")
        lr_acc = accuracy_score(y_test, lr_test_pred)
        lr_f1 = f1_score(y_test, lr_test_pred, average="macro")

        print(f"Random Forest       : accuracy={rf_acc:.3f}, macro-F1={rf_f1:.3f}")
        print(f"Logistic Regression : accuracy={lr_acc:.3f}, macro-F1={lr_f1:.3f}")

        metadata = metadata.copy()
        metadata["importance"] = rf.feature_importances_

        channel_importance = metadata.groupby("channel")["importance"].sum()
        dilation_importance = metadata.groupby("dilation")["importance"].sum()

        for channel in CHANNEL_NAMES:
            channel_rows.append({
                "fold": fold,
                "channel": channel,
                "importance": float(channel_importance.get(channel, 0.0)),
            })

        for dilation, importance in dilation_importance.items():
            dilation_rows.append({
                "fold": fold,
                "dilation": int(dilation),
                "importance": float(importance),
            })

        fold_rows.append({
            "fold": fold,
            "n_features": Z_train.shape[1],
            "dilations": ",".join(map(str, dilations)),
            "rf_test_accuracy": rf_acc,
            "rf_macro_f1": rf_f1,
            "lr_test_accuracy": lr_acc,
            "lr_macro_f1": lr_f1,
            "test_participants": ",".join(map(str, test_participants)),
        })

    folds = pd.DataFrame(fold_rows)
    channels = pd.DataFrame(channel_rows)
    dilations = pd.DataFrame(dilation_rows)

    summary = pd.DataFrame({
        "model": ["Random Forest", "Logistic Regression"],
        "mean_accuracy": [folds["rf_test_accuracy"].mean(), folds["lr_test_accuracy"].mean()],
        "std_accuracy": [folds["rf_test_accuracy"].std(), folds["lr_test_accuracy"].std()],
        "mean_macro_f1": [folds["rf_macro_f1"].mean(), folds["lr_macro_f1"].mean()],
        "std_macro_f1": [folds["rf_macro_f1"].std(), folds["lr_macro_f1"].std()],
    })

    channel_summary = channels.groupby("channel")["importance"].agg(["mean", "std"]).reset_index()
    channel_summary.columns = ["channel", "mean_importance", "std_importance"]
    channel_summary = channel_summary.sort_values("mean_importance", ascending=False)

    dilation_summary = dilations.groupby("dilation")["importance"].agg(["mean", "std"]).reset_index()
    dilation_summary.columns = ["dilation", "mean_importance", "std_importance"]
    dilation_summary = dilation_summary.sort_values("dilation")

    folds.to_csv(RESULTS_DIR / "main_cv_fold_results.csv", index=False)
    summary.to_csv(RESULTS_DIR / "main_cv_summary.csv", index=False)
    channel_summary.to_csv(RESULTS_DIR / "task_1_1_channel_importance.csv", index=False)
    dilation_summary.to_csv(RESULTS_DIR / "task_1_2_dilation_importance.csv", index=False)

    print("\n" + "=" * 70)
    print("CLASSIFICATION PERFORMANCE")
    print("=" * 70)

    for _, row in summary.iterrows():
        print(row["model"])
        print(f"  Accuracy : {row['mean_accuracy']:.4f} ± {row['std_accuracy']:.4f}")
        print(f"  Macro-F1 : {row['mean_macro_f1']:.4f} ± {row['std_macro_f1']:.4f}")

    print("\n" + "=" * 70)
    print("TASK 1.1 - SENSOR CHANNEL CONTRIBUTION")
    print("=" * 70)
    print(f"{'Channel':<22} {'Mean importance':>16} {'Std. dev.':>12}")

    for _, row in channel_summary.iterrows():
        print(f"{row['channel']:<22} {row['mean_importance']:>16.6f} {row['std_importance']:>12.6f}")

    print("\n" + "=" * 70)
    print("TASK 1.2 - TEMPORAL SCALE CONTRIBUTION")
    print("=" * 70)
    print(f"{'Dilation':<12} {'Mean importance':>16} {'Std. dev.':>12}")

    for _, row in dilation_summary.iterrows():
        print(f"{int(row['dilation']):<12} {row['mean_importance']:>16.6f} {row['std_importance']:>12.6f}")

    channel_plot = channel_summary.sort_values("mean_importance")

    plt.figure(figsize=(8, 6))
    plt.barh(channel_plot["channel"], channel_plot["mean_importance"], xerr=channel_plot["std_importance"], capsize=3)
    plt.xlabel("Mean Random Forest feature importance")
    plt.ylabel("Sensor channel")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "task_1_1_channel_importance.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.bar(dilation_summary["dilation"].astype(str), dilation_summary["mean_importance"], yerr=dilation_summary["std_importance"], capsize=3)
    plt.xlabel("Dilation")
    plt.ylabel("Mean Random Forest feature importance")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "task_1_2_dilation_importance.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("\n" + "=" * 70)
    print("FILES CREATED")
    print("=" * 70)
    print("results/main_cv_fold_results.csv")
    print("results/main_cv_summary.csv")
    print("results/task_1_1_channel_importance.csv")
    print("results/task_1_2_dilation_importance.csv")
    print("figures/task_1_1_channel_importance.png")
    print("figures/task_1_2_dilation_importance.png")


if __name__ == "__main__":
    main()