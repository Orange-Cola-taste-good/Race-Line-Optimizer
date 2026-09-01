#!/usr/bin/env python3
"""
Raceline Optimizer — main.py
All Python: FastAPI server + data pipeline + ML model + physics refinement + vision.

To train the model:
    python main.py --train

To start the API server:
    uvicorn main:app --reload --port 8000

Data layout expected:
    data/tracks/<TrackName>.csv     — columns: x_m, y_m, w_tr_right_m, w_tr_left_m
    data/racelines/<TrackName>.csv  — columns: x_m, y_m
"""

from __future__ import annotations

import argparse
import base64
import os
import pickle
import sys
import urllib.request
from pathlib import Path
from typing import Optional

import threading

import cv2
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# ─────────────────────────────────────────────────────────────────────────────
# App & global config
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(title="Raceline Optimizer API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR  = Path(os.getenv("DATA_DIR",  "data"))
MODEL_DIR = Path(os.getenv("MODEL_DIR", "models"))

N_PTS    = 500    # resample resolution for every track and raceline
N_AUG    = 10     # augmented versions per track during training
HOLD_OUT = {"Spa", "Monza", "Montreal", "Norisring", "Austin"}  # test split

SAM_URL  = "https://huggingface.co/dhkim2810/MobileSAM/resolve/main/mobile_sam.pt"

# Populated in the background after startup — check READY / use guard() before access
TRACKS: dict[str, dict] = {}
MODEL:  Optional[RandomForestRegressor] = None
SAM    = None  # lazy-loaded on first /predict/photo request
READY  = False   # flips true once tracks + model finish loading in the background


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI startup
# ─────────────────────────────────────────────────────────────────────────────
#
# Loading the tracks + (potentially training) the RandomForest can take a
# while. If that work ran directly in the startup event, uvicorn would never
# bind the port until it finished — and on a slow/small instance Render's
# port scan can time out first, making the service look dead. So we bind the
# port immediately and do the heavy loading in a background thread instead.

def _load_everything() -> None:
    global READY
    MODEL_DIR.mkdir(exist_ok=True)
    load_all_tracks()
    load_or_train_model()
    READY = True
    print("[startup] Ready — tracks and model loaded.")


@app.on_event("startup")
async def startup() -> None:
    threading.Thread(target=_load_everything, daemon=True).start()


def _require_ready() -> None:
    if not READY:
        raise HTTPException(status_code=503, detail="Server is still loading tracks/model — try again shortly.")


@app.get("/health")
def health() -> dict:
    return {"ready": READY}


# ═════════════════════════════════════════════════════════════════════════════
# DATA PIPELINE
# ═════════════════════════════════════════════════════════════════════════════

def load_csv(path: Path) -> np.ndarray:
    """Read a track or raceline CSV (comment header skipped); return float64 array."""
    return pd.read_csv(path, comment="#", header=None).values.astype(np.float64)


def arc_lengths(pts: np.ndarray) -> np.ndarray:
    """Cumulative Euclidean arc-length along a point sequence. Shape: (n,)."""
    diff = np.diff(pts, axis=0)
    seg  = np.sqrt((diff ** 2).sum(axis=1))
    return np.concatenate([[0.0], np.cumsum(seg)])


def _close(pts: np.ndarray) -> np.ndarray:
    """Append first point to close a loop (required for periodic spline)."""
    return np.vstack([pts, pts[0]])


def resample_xy(pts: np.ndarray, n: int = N_PTS) -> np.ndarray:
    """
    Cubic-spline resample a closed-loop xy sequence to exactly n
    evenly arc-length-spaced points. Returns shape (n, 2).
    """
    closed = _close(pts)
    cum    = arc_lengths(closed)
    total  = cum[-1]

    cs_x = CubicSpline(cum, closed[:, 0], bc_type="periodic")
    cs_y = CubicSpline(cum, closed[:, 1], bc_type="periodic")

    t = np.linspace(0.0, total, n, endpoint=False)
    return np.column_stack([cs_x(t), cs_y(t)])


def resample_track(raw: np.ndarray, n: int = N_PTS) -> tuple[np.ndarray, np.ndarray]:
    """
    Resample a 4-column track array (x, y, w_right, w_left) to n points.
    Returns (centerline shape (n,2), widths shape (n,2)).
    """
    pts    = raw[:, :2]
    closed = _close(raw)          # shape (len+1, 4)
    cum    = arc_lengths(closed[:, :2])
    total  = cum[-1]
    t      = np.linspace(0.0, total, n, endpoint=False)

    cs_x  = CubicSpline(cum, closed[:, 0], bc_type="periodic")
    cs_y  = CubicSpline(cum, closed[:, 1], bc_type="periodic")
    cs_wr = CubicSpline(cum, closed[:, 2], bc_type="periodic")
    cs_wl = CubicSpline(cum, closed[:, 3], bc_type="periodic")

    centerline = np.column_stack([cs_x(t), cs_y(t)])
    widths     = np.column_stack([cs_wr(t), cs_wl(t)])
    widths     = np.abs(widths)   # enforce non-negative
    return centerline, widths


def load_all_tracks() -> None:
    """Populate global TRACKS dict from data/tracks/ and data/racelines/ CSVs."""
    global TRACKS
    TRACKS = {}
    track_dir    = DATA_DIR / "tracks"
    raceline_dir = DATA_DIR / "racelines"

    for csv_path in sorted(track_dir.glob("*.csv")):
        name    = csv_path.stem
        rl_path = raceline_dir / f"{name}.csv"
        if not rl_path.exists():
            print(f"[data] Warning: no raceline for {name}, skipping.")
            continue

        track_raw    = load_csv(csv_path)
        raceline_raw = load_csv(rl_path)

        centerline, widths = resample_track(track_raw, N_PTS)
        raceline           = resample_xy(raceline_raw[:, :2], N_PTS)

        TRACKS[name] = {
            "centerline": centerline,   # (N_PTS, 2) metres
            "widths":     widths,       # (N_PTS, 2) [w_right, w_left] metres
            "raceline":   raceline,     # (N_PTS, 2) metres — ground truth
        }

    if not TRACKS:
        sys.exit(
            f"[data] ERROR: No tracks found in {track_dir}.\n"
            "       Copy *.csv files from the dataset into data/tracks/ and data/racelines/."
        )
    print(f"[data] Loaded {len(TRACKS)} tracks: {', '.join(sorted(TRACKS)[:5])}{'…' if len(TRACKS) > 5 else ''}")


# ═════════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ═════════════════════════════════════════════════════════════════════════════

def unit_tangents(pts: np.ndarray) -> np.ndarray:
    """Periodic finite-difference unit tangent at each point. Shape: (n, 2)."""
    fwd  = np.roll(pts, -1, axis=0) - np.roll(pts, 1, axis=0)
    norms = np.linalg.norm(fwd, axis=1, keepdims=True)
    return fwd / (norms + 1e-12)


def unit_normals(pts: np.ndarray) -> np.ndarray:
    """Left-hand unit normal (tangent rotated 90° CCW). Shape: (n, 2)."""
    t = unit_tangents(pts)
    return np.column_stack([-t[:, 1], t[:, 0]])


def signed_curvature(pts: np.ndarray) -> np.ndarray:
    """
    Signed curvature κ at each point using finite differences.
    Positive = left turn, negative = right turn.
    """
    dx  = np.gradient(pts[:, 0])
    dy  = np.gradient(pts[:, 1])
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    kappa = (dx * ddy - dy * ddx) / (dx ** 2 + dy ** 2 + 1e-12) ** 1.5
    return kappa


def compute_features(centerline: np.ndarray, widths: np.ndarray) -> np.ndarray:
    """
    Build ML input features per centerline point.
    Returns shape (n, 5): [x_norm, y_norm, w_right_norm, w_left_norm, kappa_scaled]

    Normalised within-track for rotation/scale invariance:
    - x, y → divided by track bounding-box diagonal
    - widths → divided by the same scale
    - κ → multiplied by scale (so κ·scale is dimensionless)
    """
    lo, hi = centerline.min(axis=0), centerline.max(axis=0)
    scale  = float(np.linalg.norm(hi - lo)) + 1e-8
    center = (lo + hi) / 2.0

    xy_norm = (centerline - center) / scale   # (n, 2)  range ≈ [-0.7, 0.7]
    w_norm  = widths / scale                  # (n, 2)  unitless
    kappa   = signed_curvature(centerline)
    k_scaled = np.clip(kappa * scale, -3.0, 3.0)[:, np.newaxis]   # (n, 1)

    return np.hstack([xy_norm, w_norm, k_scaled])   # (n, 5)


def lateral_offsets(centerline: np.ndarray, raceline: np.ndarray) -> np.ndarray:
    """
    Signed lateral offset (metres) of the raceline from the centerline.
    Positive = left of the centerline (in driving direction).
    """
    diff    = raceline - centerline           # (n, 2)
    normals = unit_normals(centerline)        # (n, 2)
    return (diff * normals).sum(axis=1)       # (n,)


def offsets_to_raceline(centerline: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """Reconstruct (x, y) raceline from centerline + signed lateral offsets."""
    normals = unit_normals(centerline)
    return centerline + offsets[:, np.newaxis] * normals


# ═════════════════════════════════════════════════════════════════════════════
# DATA AUGMENTATION
# ═════════════════════════════════════════════════════════════════════════════

def augment_one(
    centerline: np.ndarray,
    raceline:   np.ndarray,
    widths:     np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply one random augmentation:
    - random rotation (0–360°)
    - uniform scale ±10%
    - random x-reflection (50% chance; swaps w_right/w_left accordingly)
    """
    # Random rotation
    angle = np.random.uniform(0.0, 2.0 * np.pi)
    R     = np.array([[np.cos(angle), -np.sin(angle)],
                      [np.sin(angle),  np.cos(angle)]])
    c = centerline @ R.T
    r = raceline   @ R.T

    # Scale ±10%
    scale = np.random.uniform(0.9, 1.1)
    c *= scale
    r *= scale
    w  = widths * scale   # widths scale with the track

    # Reflection
    if np.random.random() > 0.5:
        c[:, 0] *= -1
        r[:, 0] *= -1
        w = w[:, ::-1]   # swap right/left width columns

    return c, r, w


def build_dataset(
    track_names: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build (X, y) arrays for training or evaluation.
    X shape: (len(names) * (1+N_AUG) * N_PTS, 5)
    y shape: (len(names) * (1+N_AUG) * N_PTS,)
    """
    X_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []

    for name in track_names:
        t          = TRACKS[name]
        centerline = t["centerline"]
        raceline   = t["raceline"]
        widths     = t["widths"]

        # Original sample
        X_list.append(compute_features(centerline, widths))
        y_list.append(lateral_offsets(centerline, raceline))

        # Augmented samples
        for _ in range(N_AUG):
            c, r, w = augment_one(centerline, raceline, widths)
            X_list.append(compute_features(c, w))
            y_list.append(lateral_offsets(c, r))

    return np.vstack(X_list), np.concatenate(y_list)


# ═════════════════════════════════════════════════════════════════════════════
# ML MODEL — RandomForestRegressor
# ═════════════════════════════════════════════════════════════════════════════

def train_model() -> RandomForestRegressor:
    """
    Train a RandomForestRegressor on the 20-track training split.
    Saves weights to models/raceline_model.pkl.
    Prints hold-out MAE in metres.
    """
    global MODEL

    train_names = [n for n in sorted(TRACKS) if n not in HOLD_OUT]
    test_names  = [n for n in sorted(TRACKS) if n     in HOLD_OUT]

    print(f"[train] Train: {len(train_names)} tracks  |  Hold-out: {len(test_names)} tracks")
    print(f"[train] Building dataset ({N_AUG + 1} versions × {N_PTS} pts each)…")

    X_train, y_train = build_dataset(train_names)
    X_test,  y_test  = build_dataset(test_names)

    print(f"[train] X_train {X_train.shape}  X_test {X_test.shape}")

    model = RandomForestRegressor(
        n_estimators   = 200,
        max_depth      = 14,
        min_samples_leaf= 4,
        max_features   = "sqrt",
        n_jobs         = -1,        # use all CPU cores
        random_state   = 42,
    )

    print("[train] Fitting RandomForestRegressor (n_estimators=200, max_depth=14)…")
    model.fit(X_train, y_train)

    # Evaluate on hold-out (original, no augmentation)
    X_eval = np.vstack([compute_features(TRACKS[n]["centerline"], TRACKS[n]["widths"])
                        for n in test_names])
    y_eval = np.concatenate([lateral_offsets(TRACKS[n]["centerline"], TRACKS[n]["raceline"])
                              for n in test_names])
    y_pred = model.predict(X_eval)
    mae    = mean_absolute_error(y_eval, y_pred)

    print(f"[eval]  Hold-out MAE: {mae:.4f} m  (target: < 1.0 m)")

    # Save
    pkl_path = MODEL_DIR / "raceline_model.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(model, f)
    print(f"[save]  Model → {pkl_path}")

    MODEL = model
    return model


def load_or_train_model() -> None:
    """Load saved model from disk; train from scratch if not found."""
    global MODEL
    pkl_path = MODEL_DIR / "raceline_model.pkl"
    if pkl_path.exists():
        with open(pkl_path, "rb") as f:
            MODEL = pickle.load(f)
        print(f"[model] Loaded from {pkl_path}")
    else:
        print("[model] No saved model — training now…")
        train_model()


def predict_raceline(
    centerline: np.ndarray,
    widths:     np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run GP model → lateral offsets → (x, y) raceline.
    Returns (raceline_ml shape (n,2), offsets shape (n,)).
    """
    feats      = compute_features(centerline, widths)
    offsets_ml = MODEL.predict(feats)
    raceline_ml = offsets_to_raceline(centerline, offsets_ml)
    return raceline_ml, offsets_ml


# ═════════════════════════════════════════════════════════════════════════════
# PHYSICS REFINEMENT — minimum curvature
# ═════════════════════════════════════════════════════════════════════════════

def _curvature_variation(offsets: np.ndarray, centerline: np.ndarray) -> float:
    """
    Objective for scipy.optimize.minimize.
    Returns sum of squared first-differences of curvature (dimensionless).
    Lower = smoother raceline.
    """
    raceline = offsets_to_raceline(centerline, offsets)
    kappa    = signed_curvature(raceline)
    return float(np.sum(np.diff(kappa) ** 2))


def minimum_curvature_refine(
    raceline_init: np.ndarray,
    centerline:    np.ndarray,
    widths:        np.ndarray,
    maxiter:       int = 300,
) -> np.ndarray:
    """
    Refine the ML raceline by minimising curvature variation.
    Bounds: offset ∈ [-w_left[i], +w_right[i]] at every point.
    Uses L-BFGS-B (fast, handles bounds).
    """
    offsets_0 = lateral_offsets(centerline, raceline_init)
    w_right   = widths[:, 0]
    w_left    = widths[:, 1]
    bounds    = [(-wl, wr) for wr, wl in zip(w_right, w_left)]

    result = minimize(
        _curvature_variation,
        offsets_0,
        args=(centerline,),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": maxiter, "ftol": 1e-10, "gtol": 1e-7},
    )
    return offsets_to_raceline(centerline, result.x)


# ═════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def curvature_profile(pts: np.ndarray) -> list[dict]:
    """Return [{dist_m, kappa}] along a raceline for frontend chart display."""
    kappa = signed_curvature(pts)
    dist  = arc_lengths(pts)
    return [{"dist_m": float(dist[i]), "kappa": float(kappa[i])} for i in range(len(pts))]


def arr_to_xy(pts: np.ndarray) -> list[dict]:
    """np.ndarray (n, 2) → [{"x": float, "y": float}, …]"""
    return [{"x": float(p[0]), "y": float(p[1])} for p in pts]


def build_response(
    centerline:   np.ndarray,
    widths:       np.ndarray,
    ground_truth: Optional[np.ndarray] = None,
    extra:        dict = {},
) -> dict:
    """
    Full prediction pipeline → API response dict.
    Runs ML prediction then physics refinement.
    """
    raceline_ml, _   = predict_raceline(centerline, widths)
    raceline_refined = minimum_curvature_refine(raceline_ml, centerline, widths)

    resp: dict = {
        "raceline":    arr_to_xy(raceline_refined),   # ML + physics
        "raceline_ml": arr_to_xy(raceline_ml),        # ML only (for toggle)
        "curvature":   curvature_profile(raceline_refined),
        "method":      "ml+physics",
    }
    if ground_truth is not None:
        resp["ground_truth"] = arr_to_xy(ground_truth)
    resp.update(extra)
    return resp


# ═════════════════════════════════════════════════════════════════════════════
# VISION — SCHEMATIC PATH  (clean PNG, white background)
# ═════════════════════════════════════════════════════════════════════════════

def extract_from_schematic(image_bytes: bytes) -> dict:
    """
    Extract track centerline and widths from a clean top-down schematic.
    Works on the dataset's _raceline.png files and similar line-art images.

    Pipeline:
        1. Grayscale → binary threshold (inverts white bg → black pixels become 255)
        2. Morphological close to connect thin lines
        3. Find two largest closed contours = outer + inner track boundary
        4. Resample both to N_PTS; centerline = midpoint; widths = boundary distance
    """
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image — ensure it is a valid PNG or JPEG.")

    gray      = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, bw     = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    kernel    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    bw        = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    if len(contours) < 2:
        raise ValueError("Fewer than 2 contours found — check that the image has clear track boundaries.")

    # Take the two largest contours by area
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]
    c0 = contours[0][:, 0, :].astype(float)
    c1 = contours[1][:, 0, :].astype(float)

    # Resample both boundaries to N_PTS
    c0_r = resample_xy(c0)
    c1_r = resample_xy(c1)

    # Centerline and widths (pixel units — caller normalises by scale_km)
    centerline_px = (c0_r + c1_r) / 2.0
    widths_px     = np.linalg.norm(c0_r - c1_r, axis=1, keepdims=True) / 2.0
    widths_px     = np.hstack([widths_px, widths_px])   # symmetric split

    return {
        "centerline":  centerline_px,   # np.ndarray, pixel units
        "widths":      widths_px,       # np.ndarray, pixel units
        "pixel_units": True,
    }


def _pixel_to_metre(
    centerline_px: np.ndarray,
    widths_px:     np.ndarray,
    scale_km:      float,
) -> tuple[np.ndarray, np.ndarray]:
    """Scale pixel-unit arrays to metres using user-provided track length."""
    pixel_arc = arc_lengths(centerline_px)[-1]
    px_per_m  = pixel_arc / (scale_km * 1000.0)
    return centerline_px / px_per_m, widths_px / px_per_m


# ═════════════════════════════════════════════════════════════════════════════
# VISION — REAL PHOTO PATH  (MobileSAM)
# ═════════════════════════════════════════════════════════════════════════════

def download_sam_if_missing() -> None:
    """Auto-download MobileSAM checkpoint (~40 MB) on first use."""
    sam_path = MODEL_DIR / "mobile_sam.pt"
    if sam_path.exists():
        return
    print(f"[sam] Downloading MobileSAM checkpoint → {sam_path} …")
    MODEL_DIR.mkdir(exist_ok=True)
    urllib.request.urlretrieve(SAM_URL, sam_path)
    mb = sam_path.stat().st_size / 1e6
    print(f"[sam] Download complete ({mb:.1f} MB)")


def load_sam():
    """Load MobileSAM ViT-T. Raises RuntimeError if mobile-sam not installed."""
    try:
        from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator
    except ImportError:
        raise RuntimeError(
            "MobileSAM not installed. Run: pip install mobile-sam"
        )
    sam_path = MODEL_DIR / "mobile_sam.pt"
    if not sam_path.exists():
        download_sam_if_missing()
    sam = sam_model_registry["vit_t"](checkpoint=str(sam_path))
    sam.eval()
    return SamAutomaticMaskGenerator(sam)


def filter_track_mask(masks: list, img: np.ndarray) -> np.ndarray:
    """
    Score each SAM candidate mask and return the one most likely to be a racetrack.

    Scoring criteria (higher = more track-like):
    - Area fraction 5–60% of image total
    - Elongated (not square, not a thin line)
    - Single large connected component (closed loop)
    """
    H, W        = img.shape[:2]
    total_area  = H * W
    best_mask   = None
    best_score  = -1.0

    for m in masks:
        seg  = m["segmentation"].astype(np.uint8)
        frac = m["area"] / total_area

        # Gross area filter
        if not (0.05 < frac < 0.65):
            continue

        # Check for a single large connected component
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(seg)
        if num_labels < 2:          # 0 = background, 1+ = components
            continue
        component_areas = stats[1:, cv2.CC_STAT_AREA]
        largest = component_areas.max()
        if largest < 0.8 * m["area"]:
            continue                # fragmented mask — skip

        # Bounding-box aspect ratio: prefer elongated shapes
        cnts, _ = cv2.findContours(seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        cnt  = max(cnts, key=cv2.contourArea)
        rect = cv2.minAreaRect(cnt)
        w_r, h_r = sorted(rect[1])
        if h_r < 1:
            continue
        aspect = w_r / h_r          # 0 = line, 1 = square

        # Score: reward mid-range aspect (not too square, not too thin)
        shape_score = 1.0 - abs(aspect - 0.35)   # peaks around aspect ≈ 0.35
        score       = frac * shape_score

        if score > best_score:
            best_score = score
            best_mask  = seg

    if best_mask is None:
        raise ValueError(
            "No track-like region detected in the photo. "
            "Try a clearer aerial/satellite view, or adjust scale_km."
        )
    return best_mask


def mask_to_boundaries(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Derive outer and inner track boundary contours from a filled track mask.
    Uses morphological dilation/erosion to separate the two edges.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    outer  = cv2.dilate(mask, kernel, iterations=4)
    inner  = cv2.erode(mask,  kernel, iterations=4)

    outer_edge = (outer - mask).astype(np.uint8)
    inner_edge = (mask  - inner).clip(0).astype(np.uint8)

    outer_cnts, _ = cv2.findContours(outer_edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    inner_cnts, _ = cv2.findContours(inner_edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if not outer_cnts or not inner_cnts:
        raise ValueError("Could not extract track boundaries from the SAM mask.")

    outer_c = max(outer_cnts, key=cv2.contourArea)[:, 0, :].astype(float)
    inner_c = max(inner_cnts, key=cv2.contourArea)[:, 0, :].astype(float)
    return outer_c, inner_c


def extract_from_photo(image_bytes: bytes, scale_km: float) -> dict:
    """
    Full real-photo pipeline:
        1. MobileSAM generates candidate masks
        2. Filter to the track-like mask
        3. Extract inner + outer boundaries
        4. Resample; compute centerline and widths (pixel units)
        5. Scale to metres using scale_km
        6. Return mask preview as base64 PNG for UI confirmation step
    """
    global SAM
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image.")

    # Lazy-load SAM
    if SAM is None:
        SAM = load_sam()

    masks      = SAM.generate(img)
    track_mask = filter_track_mask(masks, img)
    outer_px, inner_px = mask_to_boundaries(track_mask)

    outer_r    = resample_xy(outer_px)
    inner_r    = resample_xy(inner_px)

    centerline_px = (outer_r + inner_r) / 2.0
    widths_raw    = np.linalg.norm(outer_r - inner_r, axis=1, keepdims=True) / 2.0
    widths_px     = np.hstack([widths_raw, widths_raw])

    # Pixel → metre
    centerline_m, widths_m = _pixel_to_metre(centerline_px, widths_px, scale_km)

    # Mask preview (binary → PNG → base64)
    _, buf   = cv2.imencode(".png", track_mask * 255)
    mask_b64 = base64.b64encode(buf.tobytes()).decode()

    return {
        "centerline":   centerline_m,
        "widths":       widths_m,
        "mask_preview": mask_b64,
    }


# ═════════════════════════════════════════════════════════════════════════════
# API ROUTES
# ═════════════════════════════════════════════════════════════════════════════

@app.get("/tracks", response_model=list)
def list_tracks() -> list[str]:
    """Return sorted list of all loaded track names."""
    _require_ready()
    return sorted(TRACKS.keys())


@app.get("/tracks/{name}")
def get_track(name: str) -> dict:
    """Return resampled centerline and widths for a named track."""
    _require_ready()
    if name not in TRACKS:
        raise HTTPException(status_code=404, detail=f"Track '{name}' not found.")
    t = TRACKS[name]
    return {
        "name":       name,
        "centerline": arr_to_xy(t["centerline"]),
        "widths":     t["widths"].tolist(),
        "n_pts":      N_PTS,
    }


@app.post("/predict")
def predict(body: dict) -> dict:
    """
    Predict optimal raceline for a known track by name or raw CSV string.

    Body: {"name": "Spa"}
      or: {"csv": "<raw 4-column CSV text>"}
    """
    _require_ready()
    if "name" in body:
        name = body["name"]
        if name not in TRACKS:
            raise HTTPException(404, f"Track '{name}' not found.")
        t = TRACKS[name]
        return build_response(
            t["centerline"],
            t["widths"],
            ground_truth=t["raceline"],
            extra={"track": name},
        )

    if "csv" in body:
        import io
        try:
            raw = np.genfromtxt(io.StringIO(body["csv"]),
                                delimiter=",", comments="#")
            if raw.ndim != 2 or raw.shape[1] < 4:
                raise ValueError("CSV must have 4 columns: x_m, y_m, w_tr_right_m, w_tr_left_m")
            centerline, widths = resample_track(raw)
        except Exception as e:
            raise HTTPException(400, f"CSV parse error: {e}")
        return build_response(centerline, widths, extra={"track": "custom_csv"})

    raise HTTPException(400, "Request body must contain 'name' or 'csv'.")


@app.post("/predict/image")
async def predict_image(file: UploadFile = File(...), scale_km: float = 5.0) -> dict:
    """
    Predict raceline from a clean top-down schematic PNG.
    Accepts the dataset's _raceline.png format and similar line-art images.
    """
    _require_ready()
    image_bytes = await file.read()
    try:
        extracted = extract_from_schematic(image_bytes)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    centerline_px = extracted["centerline"]
    widths_px     = extracted["widths"]
    centerline_m, widths_m = _pixel_to_metre(centerline_px, widths_px, scale_km)

    return build_response(
        centerline_m, widths_m,
        extra={"source": "schematic", "scale_km": scale_km},
    )


@app.post("/predict/photo")
async def predict_photo(
    file:     UploadFile = File(...),
    scale_km: float = 5.0,
) -> dict:
    """
    Predict raceline from a real aerial or satellite photo.
    MobileSAM segments the track surface; user provides approximate track length
    (scale_km) for pixel-to-metre calibration.

    Returns sam_mask_preview (base64 PNG) alongside the raceline so the frontend
    can show a confirmation step before committing the result.
    """
    _require_ready()
    image_bytes = await file.read()
    try:
        extracted = extract_from_photo(image_bytes, scale_km)
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))

    centerline_m = np.array(extracted["centerline"])
    widths_m     = np.array(extracted["widths"])

    return build_response(
        centerline_m, widths_m,
        extra={
            "source":           "aerial_photo",
            "scale_km":         scale_km,
            "sam_mask_preview": extracted["mask_preview"],
        },
    )


# ═════════════════════════════════════════════════════════════════════════════
# STATIC FRONTEND — serve the built React app from this same service
# ═════════════════════════════════════════════════════════════════════════════
#
# Registered last so it never shadows the /tracks, /predict, /health API
# routes above. html=True serves index.html for unmatched paths (SPA routing).

_FRONTEND_BUILD = Path(__file__).parent / "frontend" / "build"
if _FRONTEND_BUILD.is_dir():
    app.mount("/", StaticFiles(directory=str(_FRONTEND_BUILD), html=True), name="frontend")
else:
    print(f"[startup] Warning: {_FRONTEND_BUILD} not found — did the frontend build run? "
          "API routes will still work, but no UI will be served at /.")


# ═════════════════════════════════════════════════════════════════════════════
# Standalone training entry point
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Raceline Optimizer — training script")
    parser.add_argument(
        "--train", action="store_true",
        help="Train the model and save to models/raceline_model.pkl"
    )
    parser.add_argument(
        "--data-dir", default="data",
        help="Path to data directory (default: data/)"
    )
    parser.add_argument(
        "--model-dir", default="models",
        help="Path to save model weights (default: models/)"
    )
    args = parser.parse_args()

    DATA_DIR  = Path(args.data_dir)
    MODEL_DIR = Path(args.model_dir)
    MODEL_DIR.mkdir(exist_ok=True)

    print("=" * 60)
    print("  Raceline Optimizer — Training Pipeline")
    print("=" * 60)

    load_all_tracks()
    train_model()

    print("\n" + "=" * 60)
    print("  Done. Next steps:")
    print("  1. Start the API:  uvicorn main:app --reload --port 8000")
    print("  2. Check docs at:  http://localhost:8000/docs")
    print("=" * 60)
