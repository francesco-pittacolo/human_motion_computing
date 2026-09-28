from pathlib import Path
import re

import numpy as np
import pandas as pd


CHANNEL_NAMES = [
    "accelerometer_x", "accelerometer_y", "accelerometer_z",
    "gravity_x", "gravity_y", "gravity_z",
    "gyros_x", "gyros_y", "gyros_z",
    "lin_accel_x", "lin_accel_y", "lin_accel_z",
    "game_rot_vec_x", "game_rot_vec_y", "game_rot_vec_z",
    "magn_field_x", "magn_field_y", "magn_field_z",
]

MOVEMENT_NAMES = {
    0: "Absence of movement",
    1: "Upward Block",
    2: "Hammering Inward Block",
    3: "Extended Outward Block",
    4: "Outward Downward Block",
    5: "Rear Elbow Block",
}

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent


DATASET_DIR = PROJECT_DIR / "KSAS-Dataset"


def find_recording_files():
    pattern = re.compile(r"^[0-5]-\d+-[id]\.csv$")
    files = [path for path in DATASET_DIR.rglob("*.csv") if pattern.match(path.name)]
    return sorted(files, key=lambda path: (int(path.stem.split("-")[0]), int(path.stem.split("-")[1]), path.stem.split("-")[2]))


def read_recording(file_path):
    df = pd.read_csv(file_path)
    values = df.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)

    if values.shape[1] != len(CHANNEL_NAMES):
        raise ValueError(f"{file_path.name}: expected 18 sensor channels, found {values.shape[1]}")
    if np.isnan(values).any():
        raise ValueError(f"{file_path.name}: missing or non-numeric sensor values found")

    return values.T


def load_ksas():
    recordings = []
    labels = []
    participants = []
    arms = []
    filenames = []

    files = find_recording_files()
    if not files:
        raise RuntimeError(f"No KSAS recordings found inside {DATASET_DIR}")

    for file_path in files:
        movement, participant, arm = file_path.stem.split("-")
        recordings.append(read_recording(file_path))
        labels.append(int(movement))
        participants.append(int(participant))
        arms.append(arm)
        filenames.append(file_path.name)

    return (
        recordings,
        np.asarray(labels, dtype=np.int64),
        np.asarray(participants, dtype=np.int64),
        np.asarray(arms),
        np.asarray(filenames),
    )


def main():
    X, y, participants, arms, filenames = load_ksas()
    lengths = np.asarray([recording.shape[1] for recording in X])

    print("=" * 70)
    print("KSAS DATASET CHECK")
    print("=" * 70)
    print("Dataset directory:", DATASET_DIR)
    print("Recordings:", len(X))
    print("Participants:", len(np.unique(participants)))
    print("Classes:", sorted(np.unique(y)))
    print("Channels per recording:", X[0].shape[0])
    print("Minimum length:", lengths.min())
    print("Maximum length:", lengths.max())
    print(f"Mean length: {lengths.mean():.3f}")
    print("Arms:", dict(zip(*np.unique(arms, return_counts=True))))

    print("\nFirst recording")
    print("-" * 70)
    print("File:", filenames[0])
    print(f"Class: {y[0]} - {MOVEMENT_NAMES[int(y[0])]}")
    print("Participant:", participants[0])
    print("Arm:", arms[0])
    print("Shape (channels, time):", X[0].shape)

    if len(X) != 240:
        print("\nWARNING: expected 240 recordings.")
    if len(np.unique(participants)) != 20:
        print("WARNING: expected 20 participants.")
    if len(np.unique(y)) != 6:
        print("WARNING: expected 6 classes.")


if __name__ == "__main__":
    main()
