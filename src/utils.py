import ast

import numpy as np
import pandas as pd
import torch

from xrocket.encoder import XRocket

from load_dataset import CHANNEL_NAMES, PROJECT_DIR


RANDOM_STATE = 42
N_SPLITS = 5
KERNEL_LENGTH = 9
MAIN_MAX_KERNEL_SPAN = 33

RESULTS_DIR = PROJECT_DIR / "results"
FIGURES_DIR = PROJECT_DIR / "figures"
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)


def make_encoder(max_kernel_span=MAIN_MAX_KERNEL_SPAN):
    return XRocket(
        in_channels=len(CHANNEL_NAMES),
        max_kernel_span=max_kernel_span,
        combination_order=1,
        feature_cap=10_000,
        kernel_length=KERNEL_LENGTH,
        max_dilations=32,
    )


def to_tensor(recording):
    return torch.tensor(recording, dtype=torch.float32).unsqueeze(0)


def fit_encoder(encoder, recording):
    encoder.fit(to_tensor(recording))

    if hasattr(encoder, "eval"):
        encoder.eval()


def encode_one(encoder, recording):
    with torch.no_grad():
        features = encoder(to_tensor(recording))

    return features.detach().cpu().numpy().reshape(-1)


def encode_dataset(encoder, recordings):
    embeddings = [encode_one(encoder, recording) for recording in recordings]
    dimensions = {len(embedding) for embedding in embeddings}

    if len(dimensions) != 1:
        raise RuntimeError(f"XROCKET returned inconsistent embedding sizes: {dimensions}")

    return np.stack(embeddings)


def channel_to_name(value):
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return value

    array = np.asarray(value)

    if array.ndim == 0:
        index = int(array.item())

        if 0 <= index < len(CHANNEL_NAMES):
            return CHANNEL_NAMES[index]

        return str(value)

    array = array.reshape(-1)

    if len(array) == len(CHANNEL_NAMES):
        active = np.flatnonzero(array)

        if len(active) == 1:
            return CHANNEL_NAMES[int(active[0])]

        if len(active) > 1:
            return " + ".join(CHANNEL_NAMES[int(index)] for index in active)

    return str(value)


def normalize_pattern(pattern):
    if isinstance(pattern, str):
        try:
            pattern = ast.literal_eval(pattern)
        except (ValueError, SyntaxError):
            return pattern

    if isinstance(pattern, np.ndarray):
        pattern = pattern.tolist()

    if isinstance(pattern, (list, tuple)):
        return str([float(value) for value in pattern])

    return str(pattern)


def feature_metadata(encoder):
    rows = []

    for pattern, dilation, channel, _ in encoder.feature_names:
        rows.append({
            "pattern": normalize_pattern(pattern),
            "dilation": int(np.asarray(dilation).reshape(-1)[0]),
            "channel": channel_to_name(channel),
        })

    return pd.DataFrame(rows)


def build_unique_feature_index(encoder):
    rows = []
    counters = {}

    for pattern, dilation, channel, _ in encoder.feature_names:
        pattern = normalize_pattern(pattern)
        dilation = int(np.asarray(dilation).reshape(-1)[0])
        channel = channel_to_name(channel)

        base_key = (pattern, dilation, channel)
        threshold_slot = counters.get(base_key, 0)
        counters[base_key] = threshold_slot + 1

        rows.append((pattern, dilation, channel, threshold_slot))

    index = pd.MultiIndex.from_tuples(
        rows,
        names=["pattern", "dilation", "channel", "threshold_slot"],
    )

    if not index.is_unique:
        raise RuntimeError("XROCKET feature index is not unique after assigning threshold slots")

    return index