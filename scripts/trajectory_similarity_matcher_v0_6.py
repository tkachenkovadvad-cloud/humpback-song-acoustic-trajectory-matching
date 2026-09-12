"""
trajectory_similarity_matcher_v0_6_resume_original_names.py

Pairwise comparison of trajectories produced by
acoustic_trajectory_v1_3_original_names_export_all_free_camera.py.

The script compares:
1. XYZ trajectory geometry in the shared PCA/UMAP space.
2. Start-aligned trajectory shape.
3. Acoustic content of event centroids.
4. Fréchet, Hausdorff, Procrustes, curvature and topology descriptors.

Two-stage workflow:
1. Fast screening of all pairs at reduced resolution.
2. Full permutation tests only for the best --top pairs.

PNG modes:
--figures strong  (default) only strong_match_candidate pairs
--figures match   strong + match candidates
--figures all     every pair
--figures none    no PNG output

Statistical meaning:
The reported p-value tests whether the observed ordered similarity is stronger
than a null distribution created by destroying the temporal order of one signal.
A low p-value supports "ordered trajectory match candidate"; it does not prove
biological identity without a labelled validation dataset.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import rankdata, gaussian_kde


N_RESAMPLE = 180
SCREEN_RESAMPLE = 60
N_PERMUTATIONS = 499
DEFAULT_TOP_CANDIDATES = 120
RANDOM_SEED = 42


def choose_folder() -> Path | None:
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        selected = filedialog.askdirectory(
            title="Select folder containing shared-space CSV outputs"
        )
        root.destroy()
        return Path(selected) if selected else None
    except Exception as exc:
        print(f"Could not open folder picker: {exc}", file=sys.stderr)
        return None


def resample_sequence(values: np.ndarray, n: int = N_RESAMPLE) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        raise ValueError("Cannot resample an empty sequence.")
    if len(values) == 1:
        return np.repeat(values, n, axis=0)

    old = np.linspace(0.0, 1.0, len(values))
    new = np.linspace(0.0, 1.0, n)
    return np.column_stack([
        np.interp(new, old, values[:, j])
        for j in range(values.shape[1])
    ])


def dtw_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Normalized multivariate dynamic-time-warping distance."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    prev = np.full(len(b) + 1, np.inf)
    prev[0] = 0.0

    for i in range(1, len(a) + 1):
        cur = np.full(len(b) + 1, np.inf)
        for j in range(1, len(b) + 1):
            cost = float(np.linalg.norm(a[i - 1] - b[j - 1]))
            cur[j] = cost + min(cur[j - 1], prev[j], prev[j - 1])
        prev = cur

    return float(prev[-1] / (len(a) + len(b)))


def path_length(xyz: np.ndarray) -> float:
    if len(xyz) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(xyz, axis=0), axis=1).sum())


def start_aligned_shape(xyz: np.ndarray) -> np.ndarray:
    xyz = np.asarray(xyz, dtype=float)
    aligned = xyz - xyz[0]
    length = path_length(aligned)
    if length > 0:
        aligned = aligned / length
    return aligned



def discrete_frechet_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Discrete Fréchet distance between two multivariate curves."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    cache = np.full((len(a), len(b)), np.nan, dtype=float)

    def solve(i: int, j: int) -> float:
        if np.isfinite(cache[i, j]):
            return float(cache[i, j])
        d = float(np.linalg.norm(a[i] - b[j]))
        if i == 0 and j == 0:
            value = d
        elif i > 0 and j == 0:
            value = max(solve(i - 1, 0), d)
        elif i == 0 and j > 0:
            value = max(solve(0, j - 1), d)
        else:
            value = max(
                min(solve(i - 1, j), solve(i - 1, j - 1), solve(i, j - 1)),
                d,
            )
        cache[i, j] = value
        return value

    return solve(len(a) - 1, len(b) - 1)


def hausdorff_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Symmetric Hausdorff distance between sampled trajectory points."""
    distances = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    return float(max(distances.min(axis=1).max(), distances.min(axis=0).max()))


def procrustes_disparity(a: np.ndarray, b: np.ndarray) -> float:
    """Residual disparity after translation, scaling and optimal rotation."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a0 = a - a.mean(axis=0, keepdims=True)
    b0 = b - b.mean(axis=0, keepdims=True)
    norm_a = float(np.linalg.norm(a0))
    norm_b = float(np.linalg.norm(b0))
    if norm_a < 1e-12 or norm_b < 1e-12:
        return np.nan
    a0 /= norm_a
    b0 /= norm_b
    u, _, vt = np.linalg.svd(b0.T @ a0, full_matrices=False)
    rotation = u @ vt
    b_aligned = b0 @ rotation
    return float(np.mean(np.sum((a0 - b_aligned) ** 2, axis=1)))


def curvature_series(xyz: np.ndarray) -> np.ndarray:
    """Turning angle between consecutive trajectory segments, in radians."""
    d = np.diff(np.asarray(xyz, dtype=float), axis=0)
    if len(d) < 2:
        return np.empty(0, dtype=float)
    left = d[:-1]
    right = d[1:]
    denom = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    valid = denom > 1e-12
    result = np.zeros(len(left), dtype=float)
    cosine = np.ones(len(left), dtype=float)
    cosine[valid] = np.sum(left[valid] * right[valid], axis=1) / denom[valid]
    result[valid] = np.arccos(np.clip(cosine[valid], -1.0, 1.0))
    return result


def curvature_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation between turning-angle profiles."""
    ca = curvature_series(a)
    cb = curvature_series(b)
    if len(ca) < 3 or len(cb) < 3:
        return np.nan
    n = min(len(ca), len(cb))
    ca = resample_sequence(ca[:, None], n).ravel()
    cb = resample_sequence(cb[:, None], n).ravel()
    if np.std(ca) < 1e-12 or np.std(cb) < 1e-12:
        return np.nan
    return float(np.corrcoef(ca, cb)[0, 1])


def count_self_intersections_2d(xy: np.ndarray) -> int:
    """Count non-adjacent segment intersections in a 2-D projection."""
    xy = np.asarray(xy, dtype=float)

    def orientation(p, q, r):
        return float((q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1]))

    count = 0
    for i in range(len(xy) - 1):
        p1, q1 = xy[i], xy[i + 1]
        for j in range(i + 2, len(xy) - 1):
            if j == i + 1:
                continue
            p2, q2 = xy[j], xy[j + 1]
            o1 = orientation(p1, q1, p2)
            o2 = orientation(p1, q1, q2)
            o3 = orientation(p2, q2, p1)
            o4 = orientation(p2, q2, q1)
            if o1 * o2 < 0 and o3 * o4 < 0:
                count += 1
    return count


def topology_descriptors(xyz: np.ndarray) -> tuple[int, int]:
    """Direction-change count and self-intersections in the first two PCA axes."""
    curve = np.asarray(xyz, dtype=float)
    curvature = curvature_series(curve)
    direction_changes = int(np.sum(curvature > np.deg2rad(45.0)))
    intersections = count_self_intersections_2d(curve[:, :2])
    return direction_changes, intersections


def topology_similarity(a: np.ndarray, b: np.ndarray) -> tuple[float, int, int, int, int]:
    """0-100 similarity based on direction changes and self-intersections."""
    turns_a, loops_a = topology_descriptors(a)
    turns_b, loops_b = topology_descriptors(b)
    turn_similarity = 1.0 - abs(turns_a - turns_b) / max(1, turns_a, turns_b)
    loop_similarity = 1.0 - abs(loops_a - loops_b) / max(1, loops_a, loops_b)
    score = 100.0 * (0.7 * turn_similarity + 0.3 * loop_similarity)
    return float(np.clip(score, 0.0, 100.0)), turns_a, turns_b, loops_a, loops_b

def chunk_shuffle(values: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Destroy global temporal order while retaining short local pieces.
    This is a stricter null than fully shuffling individual frames.
    """
    n = len(values)
    if n < 8:
        return values[rng.permutation(n)]

    chunk_size = max(3, int(round(math.sqrt(n))))
    chunks = [values[i:i + chunk_size] for i in range(0, n, chunk_size)]
    order = rng.permutation(len(chunks))
    shuffled = np.concatenate([chunks[i] for i in order], axis=0)

    if rng.random() < 0.5:
        shuffled = shuffled[::-1]
    return shuffled


def robust_standardize_pair(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = np.nan_to_num(np.asarray(a, float), nan=0.0, posinf=0.0, neginf=0.0)
    b = np.nan_to_num(np.asarray(b, float), nan=0.0, posinf=0.0, neginf=0.0)
    pooled = np.vstack([a, b])
    med = np.median(pooled, axis=0)
    q25 = np.quantile(pooled, 0.25, axis=0)
    q75 = np.quantile(pooled, 0.75, axis=0)
    scale = q75 - q25
    scale[~np.isfinite(scale) | (scale < 1e-9)] = 1.0
    return np.clip((a - med) / scale, -12.0, 12.0), np.clip((b - med) / scale, -12.0, 12.0)


def event_feature_columns(df: pd.DataFrame) -> list[str]:
    excluded_exact = {
        "source_file", "node_id",
        "start_frame", "peak_frame", "end_frame",
        "start_time_s", "peak_time_s", "end_time_s",
        "duration_s", "n_frames",
        "X", "Y", "Z",
        "peak_X", "peak_Y", "peak_Z",
        "centroid_X", "centroid_Y", "centroid_Z",
    }

    columns = []
    for col in df.columns:
        if col in excluded_exact:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            columns.append(col)
    return columns


def _normalise_signal_name(value: object) -> str:
    """Return only the original basename, preserving the .wav extension."""
    text = str(value).strip().replace("\\", "/")
    return text.rsplit("/", 1)[-1]


def _load_combined_events(folder: Path) -> dict[str, pd.DataFrame]:
    """Read the one-file export created by CSV: all events."""
    candidates = [
        folder / "all_signals_events_centroids_shared_space.csv",
        *sorted(folder.glob("*all*events*centroids*shared_space*.csv")),
    ]
    combined_file = next((p for p in candidates if p.exists()), None)
    if combined_file is None:
        return {}

    combined = pd.read_csv(combined_file)
    signal_column = next(
        (c for c in ("source_file", "signal", "signal_name", "filename", "file")
         if c in combined.columns),
        None,
    )
    if signal_column is None:
        raise RuntimeError(
            f"Found {combined_file.name}, but it has no signal-name column. "
            "Expected source_file, signal, signal_name, filename, or file."
        )

    combined = combined.copy()
    combined["__matcher_signal__"] = combined[signal_column].map(_normalise_signal_name)
    grouped = {
        str(name): group.drop(columns="__matcher_signal__").reset_index(drop=True)
        for name, group in combined.groupby("__matcher_signal__", sort=False)
    }
    print(f"Using combined event export: {combined_file.name}")
    return grouped


def _source_name_from_table(df: pd.DataFrame, fallback: str) -> str:
    for column in ("source_file", "filename", "file", "signal", "signal_name"):
        if column in df.columns and len(df):
            value = df[column].dropna()
            if len(value):
                return _normalise_signal_name(value.iloc[0])
    return fallback


def load_signals(folder: Path) -> dict[str, dict[str, pd.DataFrame]]:
    frame_files = sorted(folder.glob("*_frames_shared_space.csv"))
    if len(frame_files) < 2:
        raise RuntimeError(
            "Need at least two *_frames_shared_space.csv files in the folder."
        )

    combined_events = _load_combined_events(folder)
    signals: dict[str, dict[str, pd.DataFrame]] = {}

    for frame_file in frame_files:
        export_stem = frame_file.name.removesuffix("_frames_shared_space.csv")
        event_file = folder / f"{export_stem}_events_centroids_shared_space.csv"
        frames = pd.read_csv(frame_file)
        original_name = _source_name_from_table(frames, export_stem)

        if event_file.exists():
            events = pd.read_csv(event_file)
        elif original_name in combined_events:
            events = combined_events[original_name]
        elif export_stem in combined_events:
            events = combined_events[export_stem]
        else:
            print(
                f"Skipping {original_name}: no per-signal events and no matching rows "
                "in the combined all-signals event CSV.",
                file=sys.stderr,
            )
            continue

        required = {"X", "Y", "Z"}
        if not required.issubset(frames.columns):
            print(f"Skipping {original_name}: XYZ columns are missing.", file=sys.stderr)
            continue
        if events.empty:
            print(f"Skipping {original_name}: event table is empty.", file=sys.stderr)
            continue
        if original_name in signals:
            raise RuntimeError(f"Duplicate original filename: {original_name}")

        signals[original_name] = {
            "frames": frames,
            "events": events,
            "frame_file": frame_file,
            "event_file": event_file if event_file.exists() else None,
        }

    if len(signals) < 2:
        raise RuntimeError("Fewer than two complete signal exports were found.")

    print(f"Loaded {len(signals)} complete signals using original filenames.")
    return signals


def combined_distance(
    xyz_a: np.ndarray,
    xyz_b: np.ndarray,
    content_a: np.ndarray,
    content_b: np.ndarray,
    n_resample: int = N_RESAMPLE,
) -> tuple[float, float, float, float]:
    # Bound all expensive work by resampling first. This prevents a single long
    # selection from making hundreds of permutations take hours.
    xyz_a_r = resample_sequence(np.nan_to_num(xyz_a, nan=0.0, posinf=0.0, neginf=0.0), n_resample)
    xyz_b_r = resample_sequence(np.nan_to_num(xyz_b, nan=0.0, posinf=0.0, neginf=0.0), n_resample)

    # Real location in shared space.
    shared_a, shared_b = robust_standardize_pair(xyz_a_r, xyz_b_r)
    d_shared = dtw_distance(shared_a, shared_b)

    # Geometry after matching starts and normalizing path length.
    d_shape = dtw_distance(
        start_aligned_shape(xyz_a_r),
        start_aligned_shape(xyz_b_r),
    )

    # Ordered acoustic centroid content.
    if len(content_a) and len(content_b) and content_a.shape[1] > 0:
        content_n = min(n_resample, max(30, len(content_a), len(content_b)))
        ca_r = resample_sequence(np.nan_to_num(content_a, nan=0.0, posinf=0.0, neginf=0.0), content_n)
        cb_r = resample_sequence(np.nan_to_num(content_b, nan=0.0, posinf=0.0, neginf=0.0), content_n)
        ca, cb = robust_standardize_pair(ca_r, cb_r)
        d_content = dtw_distance(ca, cb)
    else:
        d_content = np.nan

    if np.isfinite(d_content):
        combined = 0.35 * d_shared + 0.40 * d_shape + 0.25 * d_content
    else:
        combined = 0.45 * d_shared + 0.55 * d_shape

    return d_shared, d_shape, d_content, float(combined)



def prepare_pair_arrays(
    sig_a: dict[str, pd.DataFrame],
    sig_b: dict[str, pd.DataFrame],
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray, np.ndarray]:
    xyz_a = sig_a["frames"][["X", "Y", "Z"]].to_numpy(float)
    xyz_b = sig_b["frames"][["X", "Y", "Z"]].to_numpy(float)

    common_features = sorted(
        set(event_feature_columns(sig_a["events"]))
        & set(event_feature_columns(sig_b["events"]))
    )

    content_a = (
        sig_a["events"][common_features]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(sig_a["events"][common_features].median())
        .fillna(0.0)
        .to_numpy(float)
        if common_features else np.empty((len(sig_a["events"]), 0))
    )
    content_b = (
        sig_b["events"][common_features]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(sig_b["events"][common_features].median())
        .fillna(0.0)
        .to_numpy(float)
        if common_features else np.empty((len(sig_b["events"]), 0))
    )
    return xyz_a, xyz_b, common_features, content_a, content_b


def screen_pair(
    name_a: str,
    sig_a: dict[str, pd.DataFrame],
    name_b: str,
    sig_b: dict[str, pd.DataFrame],
) -> dict[str, object]:
    xyz_a, xyz_b, common_features, content_a, content_b = prepare_pair_arrays(sig_a, sig_b)
    d_shared, d_shape, d_content, combined = combined_distance(
        xyz_a, xyz_b, content_a, content_b, n_resample=SCREEN_RESAMPLE
    )
    return {
        "signal_a": name_a,
        "signal_b": name_b,
        "screen_shared_xyz_dtw": d_shared,
        "screen_start_aligned_shape_dtw": d_shape,
        "screen_centroid_content_dtw": d_content,
        "screen_combined_distance": combined,
        "n_common_centroid_features": len(common_features),
        "n_frames_a": len(xyz_a),
        "n_frames_b": len(xyz_b),
        "n_events_a": len(content_a),
        "n_events_b": len(content_b),
    }


def compare_pair(
    name_a: str,
    sig_a: dict[str, pd.DataFrame],
    name_b: str,
    sig_b: dict[str, pd.DataFrame],
    rng: np.random.Generator,
    n_perm: int,
    max_pair_seconds: float | None = None,
) -> dict[str, object]:
    xyz_a, xyz_b, common_features, content_a, content_b = prepare_pair_arrays(sig_a, sig_b)

    d_shared, d_shape, d_content, observed = combined_distance(
        xyz_a, xyz_b, content_a, content_b
    )

    shape_a = resample_sequence(start_aligned_shape(xyz_a))
    shape_b = resample_sequence(start_aligned_shape(xyz_b))
    d_frechet = discrete_frechet_distance(shape_a, shape_b)
    d_hausdorff = hausdorff_distance(shape_a, shape_b)
    d_procrustes = procrustes_disparity(shape_a, shape_b)
    curvature_r = curvature_correlation(shape_a, shape_b)
    topology_score, turns_a, turns_b, loops_a, loops_b = topology_similarity(
        shape_a, shape_b
    )

    # Resample once before permutation testing, so every iteration has fixed cost.
    perm_xyz_a = resample_sequence(xyz_a, N_RESAMPLE)
    perm_xyz_b = resample_sequence(xyz_b, N_RESAMPLE)
    content_n = min(N_RESAMPLE, max(30, len(content_a), len(content_b))) if len(content_a) and len(content_b) else 0
    perm_content_a = resample_sequence(content_a, content_n) if content_n else content_a
    perm_content_b = resample_sequence(content_b, content_n) if content_n else content_b

    null = np.empty(n_perm, dtype=float)
    null_shared = np.empty(n_perm, dtype=float)
    null_shape = np.empty(n_perm, dtype=float)
    null_content = np.full(n_perm, np.nan, dtype=float)
    pair_clock = time.perf_counter()
    for k in range(n_perm):
        if max_pair_seconds and k % 10 == 0 and time.perf_counter() - pair_clock > max_pair_seconds:
            raise TimeoutError(f"pair exceeded {max_pair_seconds / 60.0:.1f} minutes")
        shuffled_xyz_b = chunk_shuffle(perm_xyz_b, rng)
        shuffled_content_b = (
            chunk_shuffle(perm_content_b, rng)
            if len(perm_content_b) else perm_content_b
        )
        nd_shared, nd_shape, nd_content, nd_combined = combined_distance(
            perm_xyz_a,
            shuffled_xyz_b,
            perm_content_a,
            shuffled_content_b,
        )
        null_shared[k] = nd_shared
        null_shape[k] = nd_shape
        null_content[k] = nd_content
        null[k] = nd_combined

    p_value = float((1 + np.sum(null <= observed)) / (n_perm + 1))
    null_mean = float(np.mean(null))
    null_sd = float(np.std(null, ddof=1)) if n_perm > 1 else np.nan
    effect_z = float((null_mean - observed) / null_sd) if null_sd > 0 else np.nan
    percentile_similarity = float(100.0 * np.mean(null > observed))
    shared_space_similarity = float(100.0 * np.mean(null_shared > d_shared))
    shape_similarity = float(100.0 * np.mean(null_shape > d_shape))
    finite_content_null = null_content[np.isfinite(null_content)]
    content_similarity = (
        float(100.0 * np.mean(finite_content_null > d_content))
        if np.isfinite(d_content) and len(finite_content_null)
        else np.nan
    )

    # Conservative label: p-value plus a useful effect size.
    if p_value <= 0.01 and effect_z >= 2.5:
        label = "strong_match_candidate"
    elif p_value <= 0.05 and effect_z >= 1.5:
        label = "match_candidate"
    else:
        label = "not_supported"

    row = {
        "signal_a": name_a,
        "signal_b": name_b,
        "shared_xyz_dtw": d_shared,
        "start_aligned_shape_dtw": d_shape,
        "centroid_content_dtw": d_content,
        "combined_distance": observed,
        "frechet_distance": d_frechet,
        "hausdorff_distance": d_hausdorff,
        "procrustes_disparity": d_procrustes,
        "curvature_correlation": curvature_r,
        "topology_similarity_score": topology_score,
        "direction_changes_a": turns_a,
        "direction_changes_b": turns_b,
        "self_intersections_a": loops_a,
        "self_intersections_b": loops_b,
        "shared_space_similarity_score": shared_space_similarity,
        "trajectory_shape_similarity_score": shape_similarity,
        "centroid_content_similarity_score": content_similarity,
        "null_mean": null_mean,
        "null_sd": null_sd,
        "effect_z": effect_z,
        "permutation_p": p_value,
        "similarity_percentile_vs_order_null": percentile_similarity,
        "n_common_centroid_features": len(common_features),
        "n_frames_a": len(xyz_a),
        "n_frames_b": len(xyz_b),
        "n_events_a": len(content_a),
        "n_events_b": len(content_b),
        "decision": label,
    }
    return row, null



def safe_name(text: str) -> str:
    allowed = []
    for ch in text:
        if ch.isalnum() or ch in ("-", "_"):
            allowed.append(ch)
        else:
            allowed.append("_")
    return "".join(allowed).strip("_")[:120]


def save_trajectory_overlay(
    name_a: str,
    xyz_a: np.ndarray,
    name_b: str,
    xyz_b: np.ndarray,
    out_path: Path,
) -> None:
    """Start-aligned, path-length-normalized 3D curves."""
    a = resample_sequence(start_aligned_shape(xyz_a))
    b = resample_sequence(start_aligned_shape(xyz_b))

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(a[:, 0], a[:, 1], a[:, 2], linewidth=2.4, label=name_a)
    ax.plot(b[:, 0], b[:, 1], b[:, 2], linewidth=2.4, label=name_b)

    step = max(1, len(a) // 28)
    ax.scatter(a[::step, 0], a[::step, 1], a[::step, 2], s=24, alpha=0.75)
    ax.scatter(b[::step, 0], b[::step, 1], b[::step, 2], s=24, alpha=0.75)

    ax.scatter([0], [0], [0], s=90, marker="o", label="Shared start")
    ax.set_title("Start-aligned normalized trajectory shape")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_distance_profile(
    name_a: str,
    xyz_a: np.ndarray,
    name_b: str,
    xyz_b: np.ndarray,
    out_path: Path,
) -> None:
    """Pointwise separation along normalized signal time."""
    a = resample_sequence(start_aligned_shape(xyz_a))
    b = resample_sequence(start_aligned_shape(xyz_b))
    distance = np.linalg.norm(a - b, axis=1)
    progress = np.linspace(0.0, 100.0, len(distance))

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(progress, distance, linewidth=2.2)
    ax.fill_between(progress, 0, distance, alpha=0.18)
    ax.axhline(
        float(np.median(distance)),
        linestyle="--",
        linewidth=1.4,
        label=f"Median = {np.median(distance):.4f}",
    )
    ax.set_title(f"Trajectory separation through time\n{name_a} vs {name_b}")
    ax.set_xlabel("Normalized signal progress (%)")
    ax.set_ylabel("3D separation")
    ax.set_xlim(0, 100)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_centroid_content_cloud(
    name_a: str,
    events_a: pd.DataFrame,
    name_b: str,
    events_b: pd.DataFrame,
    common_features: list[str],
    out_path: Path,
) -> None:
    """
    Project acoustic centroid content to a shared 2D PCA display.
    This figure is only for visualization; matching still uses the full feature set.
    """
    if not common_features:
        return

    a = (
        events_a[common_features]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(events_a[common_features].median())
        .fillna(0.0)
        .to_numpy(float)
    )
    b = (
        events_b[common_features]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(events_b[common_features].median())
        .fillna(0.0)
        .to_numpy(float)
    )

    pooled = np.vstack([a, b])
    med = np.nanmedian(pooled, axis=0)
    q25 = np.nanquantile(pooled, 0.25, axis=0)
    q75 = np.nanquantile(pooled, 0.75, axis=0)
    scale = q75 - q25
    scale[~np.isfinite(scale) | (scale < 1e-9)] = 1.0
    pooled = np.clip((pooled - med) / scale, -8.0, 8.0)

    # Local PCA implementation via SVD keeps this script dependency-light.
    pooled = pooled - pooled.mean(axis=0, keepdims=True)
    u, s, vt = np.linalg.svd(pooled, full_matrices=False)
    xy = pooled @ vt[:2].T

    xy_a = xy[: len(a)]
    xy_b = xy[len(a):]

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.plot(xy_a[:, 0], xy_a[:, 1], linewidth=1.5, alpha=0.65)
    ax.plot(xy_b[:, 0], xy_b[:, 1], linewidth=1.5, alpha=0.65)
    ax.scatter(xy_a[:, 0], xy_a[:, 1], s=55, alpha=0.82, label=name_a)
    ax.scatter(xy_b[:, 0], xy_b[:, 1], s=55, alpha=0.82, label=name_b)

    ax.scatter(
        [xy_a[0, 0], xy_b[0, 0]],
        [xy_a[0, 1], xy_b[0, 1]],
        s=110,
        marker="o",
        label="First event",
    )

    ax.set_title(
        f"Acoustic centroid-content cloud\n"
        f"{len(common_features)} shared features; arrows follow event order"
    )
    ax.set_xlabel("Content PC1")
    ax.set_ylabel("Content PC2")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_null_distribution(
    name_a: str,
    name_b: str,
    observed: float,
    null: np.ndarray,
    p_value: float,
    effect_z: float,
    out_path: Path,
) -> None:
    """Null distribution of distances after temporal-order destruction."""
    fig, ax = plt.subplots(figsize=(10, 5.8))

    ax.hist(null, bins=30, density=True, alpha=0.28, label="Order-destroyed null")

    if len(null) >= 10 and np.std(null) > 0:
        grid = np.linspace(float(np.min(null)), float(np.max(null)), 400)
        kde = gaussian_kde(null)
        ax.plot(grid, kde(grid), linewidth=2.2, label="Null density")

    ax.axvline(
        observed,
        linewidth=2.6,
        label=f"Observed = {observed:.4f}",
    )
    ax.axvline(
        float(np.mean(null)),
        linestyle="--",
        linewidth=1.5,
        label=f"Null mean = {np.mean(null):.4f}",
    )

    ax.set_title(
        f"Permutation evidence for ordered similarity\n"
        f"{name_a} vs {name_b} | p={p_value:.4g}, effect z={effect_z:.2f}"
    )
    ax.set_xlabel("Combined distance — smaller means more similar")
    ax.set_ylabel("Density")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)



def save_similarity_passport(
    name_a: str,
    name_b: str,
    row: dict[str, object],
    out_path: Path,
) -> None:
    """Compact summary of the pairwise similarity evidence."""
    labels = [
        "Shared acoustic space",
        "Trajectory shape",
        "Centroid content",
        "Topology",
        "Temporal-order evidence",
    ]
    values = np.array([
        float(row["shared_space_similarity_score"]),
        float(row["trajectory_shape_similarity_score"]),
        float(row["centroid_content_similarity_score"]),
        float(row["topology_similarity_score"]),
        float(row["similarity_percentile_vs_order_null"]),
    ], dtype=float)
    values = np.nan_to_num(values, nan=0.0)

    fig, ax = plt.subplots(figsize=(10, 5.8))
    y = np.arange(len(labels))
    ax.barh(y, values)
    ax.set_yticks(y, labels)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Similarity score (%)")
    ax.invert_yaxis()
    for yi, value in zip(y, values):
        ax.text(min(value + 1.2, 97), yi, f"{value:.1f}%", va="center")

    ax.set_title(
        f"Similarity passport: {name_a} vs {name_b}\n"
        f"Decision: {row['decision']} | p={float(row['permutation_p']):.4g} | "
        f"effect z={float(row['effect_z']):.2f}"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

def create_pair_figures(
    folder: Path,
    name_a: str,
    sig_a: dict[str, pd.DataFrame],
    name_b: str,
    sig_b: dict[str, pd.DataFrame],
    null: np.ndarray,
    observed: float,
    p_value: float,
    effect_z: float,
    row: dict[str, object],
) -> str:
    pair_dir = folder / "trajectory_similarity_figures"
    pair_dir.mkdir(parents=True, exist_ok=True)

    pair_stem = f"{safe_name(name_a)}__VS__{safe_name(name_b)}"

    xyz_a = sig_a["frames"][["X", "Y", "Z"]].to_numpy(float)
    xyz_b = sig_b["frames"][["X", "Y", "Z"]].to_numpy(float)

    common_features = sorted(
        set(event_feature_columns(sig_a["events"]))
        & set(event_feature_columns(sig_b["events"]))
    )

    save_trajectory_overlay(
        name_a,
        xyz_a,
        name_b,
        xyz_b,
        pair_dir / f"{pair_stem}__01_trajectory_overlay.png",
    )
    save_distance_profile(
        name_a,
        xyz_a,
        name_b,
        xyz_b,
        pair_dir / f"{pair_stem}__02_distance_profile.png",
    )
    save_centroid_content_cloud(
        name_a,
        sig_a["events"],
        name_b,
        sig_b["events"],
        common_features,
        pair_dir / f"{pair_stem}__03_centroid_content_cloud.png",
    )
    save_null_distribution(
        name_a,
        name_b,
        observed,
        null,
        p_value,
        effect_z,
        pair_dir / f"{pair_stem}__04_null_distribution.png",
    )
    save_similarity_passport(
        name_a,
        name_b,
        row,
        pair_dir / f"{pair_stem}__05_similarity_passport.png",
    )

    return str(pair_dir)

def _format_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"



def pair_key(name_a: str, name_b: str) -> str:
    return "||".join(sorted((name_a, name_b)))


def save_partial(rows: list[dict[str, object]], path: Path) -> None:
    if not rows:
        return
    pd.DataFrame(rows).to_csv(path, index=False)

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "folder",
        nargs="?",
        type=Path,
        help="Folder containing shared-space CSV outputs.",
    )
    parser.add_argument(
        "--permutations",
        type=int,
        default=N_PERMUTATIONS,
        help="Number of temporal-order null permutations for shortlisted pairs.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=DEFAULT_TOP_CANDIDATES,
        help="Number of best screening pairs receiving full permutation tests.",
    )
    parser.add_argument(
        "--max-pair-minutes",
        type=float,
        default=30.0,
        help="Skip a full-test pair if it exceeds this time; 0 disables the limit.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore any partial/full results from an earlier run.",
    )
    parser.add_argument(
        "--figures",
        default="strong",
        choices=["none", "strong", "match", "all", "0", "1"],
        help="PNG output mode: none/0, strong (default), match, or all/1.",
    )
    args = parser.parse_args()
    figure_mode = {"0": "none", "1": "all"}.get(args.figures, args.figures)

    folder = args.folder or choose_folder()
    if folder is None:
        print("No folder selected.")
        return 0

    try:
        signals = load_signals(folder)
        rng = np.random.default_rng(RANDOM_SEED)
        pairs = list(combinations(sorted(signals), 2))
        total_pairs = len(pairs)
        top_n = min(max(1, args.top), total_pairs)

        print(f"\nStage 1/2: fast screening of {total_pairs} pairs")
        print(f"Screening resolution: {SCREEN_RESAMPLE} points; no permutations; no PNG")
        screening_rows = []
        stage_start = time.perf_counter()
        report_every = max(1, total_pairs // 100)

        for index, (name_a, name_b) in enumerate(pairs, start=1):
            screening_rows.append(
                screen_pair(name_a, signals[name_a], name_b, signals[name_b])
            )
            if index == 1 or index % report_every == 0 or index == total_pairs:
                elapsed = time.perf_counter() - stage_start
                rate = index / elapsed if elapsed > 0 else 0.0
                eta = (total_pairs - index) / rate if rate > 0 else 0.0
                print(
                    f"Screening: {index}/{total_pairs} "
                    f"({100.0 * index / total_pairs:5.1f}%) | "
                    f"elapsed {_format_duration(elapsed)} | ETA {_format_duration(eta)}"
                )

        screening = pd.DataFrame(screening_rows).sort_values(
            "screen_combined_distance", ascending=True
        ).reset_index(drop=True)
        screening["screen_rank"] = np.arange(1, len(screening) + 1)
        screening_out = folder / "trajectory_similarity_screening_all_pairs.csv"
        screening.to_csv(screening_out, index=False)

        selected = screening.head(top_n)
        print(f"\nStage 2/2: full permutation tests for best {top_n} pairs")
        print(f"Permutations per selected pair: {max(99, args.permutations)}")

        partial_out = folder / "trajectory_similarity_matches_partial.csv"
        final_out = folder / "trajectory_similarity_matches.csv"
        full_rows: list[dict[str, object]] = []
        completed: set[str] = set()

        if not args.no_resume:
            resume_file = final_out if final_out.exists() else partial_out
            if resume_file.exists():
                previous = pd.read_csv(resume_file)
                if {"signal_a", "signal_b"}.issubset(previous.columns):
                    full_rows = previous.to_dict("records")
                    completed = {pair_key(str(r["signal_a"]), str(r["signal_b"])) for r in full_rows}
                    print(f"Resume: loaded {len(completed)} completed full tests from {resume_file.name}")

        stage_start = time.perf_counter()
        strong_count = sum(str(r.get("decision")) == "strong_match_candidate" for r in full_rows)
        match_count = sum(str(r.get("decision")) == "match_candidate" for r in full_rows)
        done_this_run = 0
        remaining = sum(pair_key(str(c.signal_a), str(c.signal_b)) not in completed for c in selected.itertuples(index=False))

        for candidate in selected.itertuples(index=False):
            name_a = str(candidate.signal_a)
            name_b = str(candidate.signal_b)
            key = pair_key(name_a, name_b)
            if key in completed:
                continue

            pair_start = time.perf_counter()
            # A per-pair seed makes resume deterministic and independent of earlier pairs.
            seed = (RANDOM_SEED + sum((i + 1) * ord(ch) for i, ch in enumerate(key))) % (2**32 - 1)
            pair_rng = np.random.default_rng(seed)
            try:
                row, null = compare_pair(
                    name_a, signals[name_a], name_b, signals[name_b],
                    pair_rng, max(99, args.permutations),
                    None if args.max_pair_minutes <= 0 else args.max_pair_minutes * 60.0,
                )
                row["screen_rank"] = int(candidate.screen_rank)
                row["screen_combined_distance"] = float(candidate.screen_combined_distance)
                row["pair_seconds"] = float(time.perf_counter() - pair_start)
                row["error"] = ""

                decision = str(row["decision"])
                strong_count += decision == "strong_match_candidate"
                match_count += decision == "match_candidate"
                make_figures = (
                    figure_mode == "all"
                    or (figure_mode == "strong" and decision == "strong_match_candidate")
                    or (figure_mode == "match" and decision in {"strong_match_candidate", "match_candidate"})
                )
                if make_figures:
                    row["figures_folder"] = create_pair_figures(
                        folder, name_a, signals[name_a], name_b, signals[name_b],
                        null, float(row["combined_distance"]),
                        float(row["permutation_p"]), float(row["effect_z"]), row,
                    )
                else:
                    row["figures_folder"] = ""
            except Exception as exc:
                decision = "test_error"
                row = {
                    "signal_a": name_a, "signal_b": name_b,
                    "screen_rank": int(candidate.screen_rank),
                    "screen_combined_distance": float(candidate.screen_combined_distance),
                    "n_frames_a": len(signals[name_a]["frames"]),
                    "n_frames_b": len(signals[name_b]["frames"]),
                    "n_events_a": len(signals[name_a]["events"]),
                    "n_events_b": len(signals[name_b]["events"]),
                    "decision": decision,
                    "pair_seconds": float(time.perf_counter() - pair_start),
                    "error": f"{type(exc).__name__}: {exc}",
                    "figures_folder": "",
                }
                print(f"WARNING: {name_a} vs {name_b}: {row['error']}", file=sys.stderr)

            full_rows.append(row)
            completed.add(key)
            done_this_run += 1
            save_partial(full_rows, partial_out)

            elapsed = time.perf_counter() - stage_start
            rate = done_this_run / elapsed if elapsed > 0 else 0.0
            eta = (remaining - done_this_run) / rate if rate > 0 else 0.0
            print(
                f"Full tests: {len(completed)}/{top_n} | {name_a} vs {name_b} | "
                f"{decision} | pair {_format_duration(time.perf_counter()-pair_start)} | "
                f"frames {row.get('n_frames_a','?')}/{row.get('n_frames_b','?')} | "
                f"strong {strong_count}, match {match_count} | "
                f"elapsed {_format_duration(elapsed)} | ETA {_format_duration(eta)}"
            )

        result = pd.DataFrame(full_rows)
        if "permutation_p" in result.columns:
            result = result.sort_values(
                ["permutation_p", "combined_distance"], ascending=[True, True], na_position="last"
            )
        result.to_csv(final_out, index=False)
        out = final_out
        print("\nCreated:")
        print(f"  {screening_out}")
        print(f"  {out}")
        if figure_mode != "none":
            print(f"  PNG mode: {figure_mode}")
            print(f"  {folder / 'trajectory_similarity_figures'}")
        print("\nBest candidates:")
        columns = [c for c in [
            "signal_a", "signal_b", "decision", "permutation_p",
            "effect_z", "combined_distance", "screen_rank", "pair_seconds", "error",
        ] if c in result.columns]
        print(result[columns].head(20).to_string(index=False))
        return 0

    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
