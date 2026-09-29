"""
RMR spatial interpolation benchmark from synthetic drill holes
==============================================================

End-to-end research-style workflow:
1. Generate synthetic drill-hole RMR data with spatial structure.
2. Inject realistic data-quality problems (missing values, duplicates, bad RMR).
3. ETL / quality-control the raw table.
4. Perform exploratory plots and empirical variogram analysis.
5. Compare three interpolation models using spatially blocked cross-validation:
      - Ordinary Kriging (implemented from scratch, spherical variogram)
      - Gaussian Process Regression (Matérn covariance)
      - Random Forest Regression
6. Quantify point-prediction performance and uncertainty behavior.
7. Fit final models to all cleaned samples.
8. Generate RMR prediction maps, uncertainty maps, and absolute-error maps at a
   requested depth slice.
9. Save CSV results and publication-ready PNG figures.

Important interpretation note
-----------------------------
Kriging variance and GPR posterior standard deviation arise from explicit spatial
covariance models. Random-Forest "uncertainty" here is the standard deviation of
predictions across trees. It is an ensemble-spread proxy, not a formally calibrated
spatial predictive variance. The script therefore evaluates 95% interval coverage
for all three but labels RF uncertainty accordingly.

Run:
    python rmr_spatial_interpolation_comparison.py

Optional:
    python rmr_spatial_interpolation_comparison.py \
        --output-dir outputs_rmr \
        --seed 42 \
        --slice-depth 100 \
        --grid-size 80

Dependencies:
    numpy, pandas, scipy, scikit-learn, matplotlib
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.linalg import lu_factor, lu_solve
from scipy.optimize import curve_fit
from scipy.spatial.distance import cdist, pdist
from scipy.stats import qmc, spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class DomainConfig:
    x_min: float = 0.0
    x_max: float = 1000.0
    y_min: float = 0.0
    y_max: float = 800.0
    depth_min: float = 10.0
    depth_max: float = 190.0

    # Geological anisotropy scaling lengths (metres). Distances are computed in
    # dimensionless coordinates x/Lx, y/Ly, depth/Lz. In real work, estimate
    # these from directional variograms or geological knowledge.
    lx: float = 180.0
    ly: float = 140.0
    lz: float = 55.0


DOMAIN = DomainConfig()
RMR_MIN = 0.0
RMR_MAX = 100.0


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------

def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def ensure_output_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def transform_coordinates(coords: np.ndarray, domain: DomainConfig = DOMAIN) -> np.ndarray:
    """Scale XYZ coordinates to encode a simple geological anisotropy model."""
    arr = np.asarray(coords, dtype=float)
    scale = np.array([domain.lx, domain.ly, domain.lz], dtype=float)
    return arr / scale


def dataframe_coordinates(df: pd.DataFrame) -> np.ndarray:
    return df[["Easting_m", "Northing_m", "Depth_m"]].to_numpy(dtype=float)


# -----------------------------------------------------------------------------
# Synthetic geology and drill-hole generation
# -----------------------------------------------------------------------------

def true_rmr_function(x: np.ndarray, y: np.ndarray, depth: np.ndarray) -> np.ndarray:
    """
    Latent synthetic RMR field used only because this is a controlled experiment.

    The field contains:
      * a depth trend,
      * smooth spatial undulations,
      * a competent rock domain,
      * an inclined weak/fault zone,
      * a deeper weak lens.

    These ingredients deliberately create a field that is not perfectly stationary,
    giving the three model classes something non-trivial to learn.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    depth = np.asarray(depth, dtype=float)

    baseline = 67.0 - 0.035 * depth

    regional = (
        5.5 * np.sin(x / 150.0)
        + 4.5 * np.cos(y / 115.0)
        + 3.0 * np.sin((x + 0.7 * y) / 210.0)
        + 2.0 * np.cos(depth / 35.0)
    )

    competent_domain = 13.0 * np.exp(
        -0.5
        * (
            ((x - 720.0) / 170.0) ** 2
            + ((y - 245.0) / 135.0) ** 2
            + ((depth - 80.0) / 95.0) ** 2
        )
    )

    # Distance to an inclined fault trace y = 0.58*x + 95 m.
    fault_distance = (y - (0.58 * x + 95.0)) / np.sqrt(1.0 + 0.58**2)
    fault_zone = -20.0 * np.exp(-0.5 * (fault_distance / 48.0) ** 2) * (
        0.72 + 0.28 * np.exp(-0.5 * ((depth - 95.0) / 65.0) ** 2)
    )

    weak_lens = -11.0 * np.exp(
        -0.5
        * (
            ((x - 275.0) / 120.0) ** 2
            + ((y - 610.0) / 105.0) ** 2
            + ((depth - 135.0) / 45.0) ** 2
        )
    )

    rmr = baseline + regional + competent_domain + fault_zone + weak_lens
    return np.clip(rmr, 15.0, 92.0)


def assign_rock_unit(x: np.ndarray, y: np.ndarray, depth: np.ndarray) -> np.ndarray:
    """Create a simple categorical rock-unit field for realism/QC reporting."""
    x = np.asarray(x)
    y = np.asarray(y)
    depth = np.asarray(depth)

    unit = np.full(x.shape, "Granodiorite", dtype=object)
    unit[(x < 420) & (y > 430)] = "Metasediment"
    unit[(x > 620) & (y < 390)] = "Massive Diorite"
    unit[depth > 150] = np.where(unit[depth > 150] == "Granodiorite", "Altered Granodiorite", unit[depth > 150])
    return unit


def generate_synthetic_drillholes(
    seed: int = 42,
    n_holes: int = 32,
    depth_spacing: float = 20.0,
    domain: DomainConfig = DOMAIN,
) -> pd.DataFrame:
    """Generate vertical drill holes using Latin-hypercube collar locations."""
    rng = np.random.default_rng(seed)

    sampler = qmc.LatinHypercube(d=2, seed=seed)
    uv = sampler.random(n=n_holes)
    collars = qmc.scale(
        uv,
        [domain.x_min + 25.0, domain.y_min + 25.0],
        [domain.x_max - 25.0, domain.y_max - 25.0],
    )

    depths = np.arange(domain.depth_min, domain.depth_max + 0.1, depth_spacing)
    rows = []

    for i, (x, y) in enumerate(collars, start=1):
        hole_id = f"DH{i:03d}"
        hole_bias = rng.normal(0.0, 1.2)

        # Slight interval-depth jitter mimics irregular logging/compositing.
        hole_depths = np.clip(depths + rng.normal(0.0, 1.0, size=len(depths)), domain.depth_min, domain.depth_max)
        latent = true_rmr_function(
            np.full_like(hole_depths, x),
            np.full_like(hole_depths, y),
            hole_depths,
        )
        observed = latent + hole_bias + rng.normal(0.0, 3.0, size=len(hole_depths))
        observed = np.clip(observed, RMR_MIN, RMR_MAX)

        units = assign_rock_unit(
            np.full_like(hole_depths, x),
            np.full_like(hole_depths, y),
            hole_depths,
        )

        for d, rmr, unit, true_rmr in zip(hole_depths, observed, units, latent):
            rows.append(
                {
                    "HoleID": hole_id,
                    "Easting_m": x,
                    "Northing_m": y,
                    "Depth_m": d,
                    "RMR": rmr,
                    "RockUnit": unit,
                    "SyntheticTrueRMR": true_rmr,
                }
            )

    df = pd.DataFrame(rows)
    return df.sort_values(["HoleID", "Depth_m"]).reset_index(drop=True)


def inject_data_quality_issues(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Create a raw table with realistic ETL problems for demonstration."""
    rng = np.random.default_rng(seed + 1000)
    raw = df.copy()
    n = len(raw)

    # Missing target values.
    miss_rmr_idx = rng.choice(raw.index, size=max(1, int(0.015 * n)), replace=False)
    raw.loc[miss_rmr_idx, "RMR"] = np.nan

    # Missing coordinate values.
    available = raw.index.difference(miss_rmr_idx)
    miss_x_idx = rng.choice(available, size=max(1, int(0.006 * n)), replace=False)
    raw.loc[miss_x_idx, "Easting_m"] = np.nan

    # Physically impossible RMR values.
    available = raw.index.difference(miss_rmr_idx).difference(miss_x_idx)
    bad_idx = rng.choice(available, size=max(2, int(0.010 * n)), replace=False)
    half = len(bad_idx) // 2
    raw.loc[bad_idx[:half], "RMR"] = -12.0
    raw.loc[bad_idx[half:], "RMR"] = 128.0

    # Convert a few numerics to strings to demonstrate numeric coercion.
    string_idx = rng.choice(raw.index, size=max(2, int(0.010 * n)), replace=False)
    raw["Depth_m"] = raw["Depth_m"].astype(object)
    for idx in string_idx:
        raw.at[idx, "Depth_m"] = f" {float(raw.at[idx, 'Depth_m']):.3f} "

    # Duplicate a small set of rows.
    dup = raw.sample(n=max(3, int(0.015 * n)), random_state=seed)
    raw = pd.concat([raw, dup], ignore_index=True)

    return raw.sample(frac=1.0, random_state=seed).reset_index(drop=True)


# -----------------------------------------------------------------------------
# ETL / data quality control
# -----------------------------------------------------------------------------

def etl_clean_rmr(raw: pd.DataFrame, domain: DomainConfig = DOMAIN) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Clean and validate the raw drill-hole RMR table."""
    df = raw.copy()
    audit: Dict[str, int] = {"raw_rows": int(len(df))}

    required = ["HoleID", "Easting_m", "Northing_m", "Depth_m", "RMR"]
    missing_columns = [c for c in required if c not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Normalize IDs / strings.
    df["HoleID"] = df["HoleID"].astype(str).str.strip()
    if "RockUnit" in df.columns:
        df["RockUnit"] = df["RockUnit"].astype(str).str.strip()

    # Numeric coercion catches strings, malformed values become NaN.
    numeric_cols = ["Easting_m", "Northing_m", "Depth_m", "RMR"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    before = len(df)
    df = df.drop_duplicates(subset=["HoleID", "Easting_m", "Northing_m", "Depth_m", "RMR"])
    audit["duplicates_removed"] = int(before - len(df))

    before = len(df)
    df = df.dropna(subset=required)
    audit["missing_required_removed"] = int(before - len(df))

    physical_mask = (
        df["Easting_m"].between(domain.x_min, domain.x_max)
        & df["Northing_m"].between(domain.y_min, domain.y_max)
        & df["Depth_m"].between(domain.depth_min - 2.0, domain.depth_max + 2.0)
        & df["RMR"].between(RMR_MIN, RMR_MAX)
    )
    before = len(df)
    df = df.loc[physical_mask].copy()
    audit["physically_invalid_removed"] = int(before - len(df))

    # Round coordinates/depth only after QC, preserving adequate precision.
    df["Easting_m"] = df["Easting_m"].round(3)
    df["Northing_m"] = df["Northing_m"].round(3)
    df["Depth_m"] = df["Depth_m"].round(3)
    df["RMR"] = df["RMR"].round(3)

    df = df.sort_values(["HoleID", "Depth_m"]).reset_index(drop=True)

    audit["final_rows"] = int(len(df))
    audit["holes_retained"] = int(df["HoleID"].nunique())

    if len(df) < 30:
        raise ValueError("Too few valid samples remain after ETL for spatial modeling.")

    return df, audit


# -----------------------------------------------------------------------------
# Variogram and Ordinary Kriging implementation
# -----------------------------------------------------------------------------

def spherical_variogram(h: np.ndarray, nugget: float, partial_sill: float, range_: float) -> np.ndarray:
    """Spherical semivariogram with gamma(0)=0 and a nugget discontinuity for h>0."""
    h = np.asarray(h, dtype=float)
    range_ = max(float(range_), 1e-8)
    hr = h / range_
    core = nugget + partial_sill * (1.5 * hr - 0.5 * hr**3)
    gamma = np.where(h <= range_, core, nugget + partial_sill)
    gamma = np.where(h <= 1e-12, 0.0, gamma)
    return gamma


def empirical_variogram(
    coords_scaled: np.ndarray,
    values: np.ndarray,
    n_lags: int = 12,
    max_distance_quantile: float = 0.80,
) -> pd.DataFrame:
    """Compute an isotropic experimental semivariogram from pairwise distances."""
    d = pdist(coords_scaled, metric="euclidean")
    dv = pdist(np.asarray(values, dtype=float).reshape(-1, 1), metric="euclidean")
    semivariance = 0.5 * dv**2

    max_dist = float(np.quantile(d, max_distance_quantile))
    bins = np.linspace(0.0, max_dist, n_lags + 1)
    rows = []

    for i in range(n_lags):
        lo, hi = bins[i], bins[i + 1]
        mask = (d > lo) & (d <= hi)
        if mask.sum() >= 10:
            rows.append(
                {
                    "lag": float(np.mean(d[mask])),
                    "semivariance": float(np.mean(semivariance[mask])),
                    "pairs": int(mask.sum()),
                }
            )

    if len(rows) < 4:
        raise RuntimeError("Insufficient populated lag bins to fit a variogram.")

    return pd.DataFrame(rows)


def fit_spherical_variogram(empirical: pd.DataFrame, values: np.ndarray) -> Tuple[float, float, float]:
    """Fit nugget, partial sill, and range by weighted nonlinear least squares."""
    h = empirical["lag"].to_numpy(dtype=float)
    g = empirical["semivariance"].to_numpy(dtype=float)
    pairs = empirical["pairs"].to_numpy(dtype=float)

    var_y = max(float(np.var(values, ddof=1)), 1.0)
    max_h = max(float(h.max()), 0.1)

    p0 = [0.08 * var_y, 0.92 * var_y, max(float(np.median(h)), 0.15)]
    lower = [0.0, 1e-6, max(0.05 * max_h, 1e-3)]
    upper = [1.5 * var_y, 3.0 * var_y, 2.5 * max_h]

    # More pair support => smaller sigma => higher fitting weight.
    sigma = 1.0 / np.sqrt(np.maximum(pairs, 1.0))

    try:
        params, _ = curve_fit(
            spherical_variogram,
            h,
            g,
            p0=p0,
            bounds=(lower, upper),
            sigma=sigma,
            absolute_sigma=False,
            maxfev=20000,
        )
    except Exception as exc:
        logging.warning("Variogram optimization failed (%s). Falling back to robust defaults.", exc)
        params = np.array([0.08 * var_y, 0.92 * var_y, max(float(np.median(h)), 0.2)])

    return tuple(float(x) for x in params)


class OrdinaryKriging3D:
    """Minimal ordinary-kriging estimator using a fitted spherical semivariogram."""

    def __init__(self, n_lags: int = 12):
        self.n_lags = n_lags
        self.coords_: Optional[np.ndarray] = None
        self.values_: Optional[np.ndarray] = None
        self.params_: Optional[Tuple[float, float, float]] = None
        self.empirical_: Optional[pd.DataFrame] = None
        self._lu = None

    def fit(self, coords_scaled: np.ndarray, values: np.ndarray) -> "OrdinaryKriging3D":
        x = np.asarray(coords_scaled, dtype=float)
        y = np.asarray(values, dtype=float)
        if x.ndim != 2 or x.shape[1] != 3:
            raise ValueError("coords_scaled must have shape (n_samples, 3).")
        if len(x) != len(y):
            raise ValueError("Coordinate and target lengths do not match.")

        self.coords_ = x
        self.values_ = y
        self.empirical_ = empirical_variogram(x, y, n_lags=self.n_lags)
        self.params_ = fit_spherical_variogram(self.empirical_, y)

        distances = cdist(x, x, metric="euclidean")
        gamma = spherical_variogram(distances, *self.params_)
        np.fill_diagonal(gamma, 0.0)

        n = len(x)
        a = np.empty((n + 1, n + 1), dtype=float)
        a[:n, :n] = gamma
        a[:n, n] = 1.0
        a[n, :n] = 1.0
        a[n, n] = 0.0

        # Tiny numerical stabilization without materially changing the kriging system.
        a[:n, :n] += np.eye(n) * 1e-10
        self._lu = lu_factor(a)
        return self

    def predict(self, coords_scaled: np.ndarray, chunk_size: int = 2500) -> Tuple[np.ndarray, np.ndarray]:
        if self.coords_ is None or self.values_ is None or self.params_ is None or self._lu is None:
            raise RuntimeError("Kriging model must be fitted before predict().")

        q = np.asarray(coords_scaled, dtype=float)
        n_train = len(self.coords_)
        means = np.empty(len(q), dtype=float)
        variances = np.empty(len(q), dtype=float)

        for start in range(0, len(q), chunk_size):
            stop = min(start + chunk_size, len(q))
            block = q[start:stop]

            d = cdist(self.coords_, block, metric="euclidean")
            gamma_to_query = spherical_variogram(d, *self.params_)

            b = np.vstack([gamma_to_query, np.ones((1, len(block)), dtype=float)])
            solution = lu_solve(self._lu, b)
            weights = solution[:n_train, :]
            mu = solution[n_train, :]

            means[start:stop] = weights.T @ self.values_
            kv = np.sum(weights * gamma_to_query, axis=0) + mu
            variances[start:stop] = np.maximum(kv, 0.0)

        return means, np.sqrt(variances)


# -----------------------------------------------------------------------------
# GPR and Random Forest wrappers
# -----------------------------------------------------------------------------

def build_gpr(seed: int = 42) -> GaussianProcessRegressor:
    """Matérn GPR with estimated signal/noise hyperparameters."""
    kernel = (
        ConstantKernel(1.0, (1e-2, 1e2))
        * Matern(length_scale=[1.0, 1.0, 1.0], length_scale_bounds=(0.08, 8.0), nu=1.5)
        + WhiteKernel(noise_level=0.08, noise_level_bounds=(1e-4, 2.0))
    )
    return GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,
        n_restarts_optimizer=0,
        random_state=seed,
    )


def build_random_forest(seed: int = 42) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=400,
        min_samples_leaf=2,
        max_features=1.0,
        bootstrap=True,
        n_jobs=-1,
        random_state=seed,
    )


def random_forest_predict_with_spread(
    model: RandomForestRegressor,
    x: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return RF mean prediction and tree-to-tree standard deviation proxy."""
    tree_predictions = np.vstack([tree.predict(x) for tree in model.estimators_])
    mean = tree_predictions.mean(axis=0)
    std = tree_predictions.std(axis=0, ddof=1)
    return mean, std


# -----------------------------------------------------------------------------
# Spatial blocked cross-validation and metrics
# -----------------------------------------------------------------------------

def make_spatial_blocks(df: pd.DataFrame, nx: int = 4, ny: int = 4) -> pd.Series:
    """Assign each collar/sample to a rectangular XY spatial block."""
    x_bins = np.linspace(DOMAIN.x_min, DOMAIN.x_max, nx + 1)
    y_bins = np.linspace(DOMAIN.y_min, DOMAIN.y_max, ny + 1)

    ix = np.clip(np.digitize(df["Easting_m"].to_numpy(), x_bins) - 1, 0, nx - 1)
    iy = np.clip(np.digitize(df["Northing_m"].to_numpy(), y_bins) - 1, 0, ny - 1)
    return pd.Series(ix + nx * iy, index=df.index, name="SpatialBlock")


def metric_row(y_true: np.ndarray, y_pred: np.ndarray, y_std: Optional[np.ndarray] = None) -> Dict[str, float]:
    err = y_pred - y_true
    result = {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
        "Bias": float(np.mean(err)),
    }

    if y_std is not None:
        s = np.maximum(np.asarray(y_std, dtype=float), 1e-9)
        lo = y_pred - 1.96 * s
        hi = y_pred + 1.96 * s
        coverage = np.mean((y_true >= lo) & (y_true <= hi))
        width = np.mean(hi - lo)
        corr = spearmanr(s, np.abs(err), nan_policy="omit").statistic
        result.update(
            {
                "Coverage95": float(coverage),
                "Mean95Width": float(width),
                "UncertaintyAbsErrorSpearman": float(corr) if np.isfinite(corr) else np.nan,
            }
        )
    else:
        result.update({"Coverage95": np.nan, "Mean95Width": np.nan, "UncertaintyAbsErrorSpearman": np.nan})

    return result


def spatial_cross_validate(df: pd.DataFrame, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Spatial block CV. Entire XY regions are held out together."""
    data = df.copy()
    data["SpatialBlock"] = make_spatial_blocks(data)

    coords = transform_coordinates(dataframe_coordinates(data))
    target = data["RMR"].to_numpy(dtype=float)
    groups = data["SpatialBlock"].to_numpy()

    unique_groups = np.unique(groups)
    n_splits = min(5, len(unique_groups))
    if n_splits < 3:
        raise RuntimeError("Need at least 3 occupied spatial blocks for blocked CV.")

    cv = GroupKFold(n_splits=n_splits)
    fold_metrics = []
    prediction_rows = []

    for fold, (train_idx, test_idx) in enumerate(cv.split(coords, target, groups), start=1):
        logging.info("Spatial CV fold %d/%d: train=%d, test=%d", fold, n_splits, len(train_idx), len(test_idx))

        x_train, x_test = coords[train_idx], coords[test_idx]
        y_train, y_test = target[train_idx], target[test_idx]

        models = {}

        # Ordinary Kriging
        ok = OrdinaryKriging3D(n_lags=12).fit(x_train, y_train)
        ok_mean, ok_std = ok.predict(x_test)
        models["Ordinary Kriging"] = (ok_mean, ok_std)

        # Gaussian Process Regression
        gpr = build_gpr(seed + fold)
        gpr.fit(x_train, y_train)
        gpr_mean, gpr_std = gpr.predict(x_test, return_std=True)
        models["Gaussian Process"] = (gpr_mean, gpr_std)

        # Random Forest
        rf = build_random_forest(seed + fold)
        rf.fit(x_train, y_train)
        rf_mean, rf_std = random_forest_predict_with_spread(rf, x_test)
        models["Random Forest"] = (rf_mean, rf_std)

        for model_name, (pred, std) in models.items():
            row = {"Fold": fold, "Model": model_name, **metric_row(y_test, pred, std)}
            fold_metrics.append(row)

            for local_i, global_i in enumerate(test_idx):
                prediction_rows.append(
                    {
                        "RowIndex": int(global_i),
                        "Fold": fold,
                        "Model": model_name,
                        "ObservedRMR": float(y_test[local_i]),
                        "PredictedRMR": float(pred[local_i]),
                        "PredStd": float(std[local_i]),
                        "Easting_m": float(data.iloc[global_i]["Easting_m"]),
                        "Northing_m": float(data.iloc[global_i]["Northing_m"]),
                        "Depth_m": float(data.iloc[global_i]["Depth_m"]),
                        "SpatialBlock": int(data.iloc[global_i]["SpatialBlock"]),
                    }
                )

    folds = pd.DataFrame(fold_metrics)
    preds = pd.DataFrame(prediction_rows)

    summary = (
        folds.groupby("Model", as_index=False)
        .agg(
            RMSE_mean=("RMSE", "mean"),
            RMSE_sd=("RMSE", "std"),
            MAE_mean=("MAE", "mean"),
            MAE_sd=("MAE", "std"),
            R2_mean=("R2", "mean"),
            R2_sd=("R2", "std"),
            Bias_mean=("Bias", "mean"),
            Coverage95_mean=("Coverage95", "mean"),
            Mean95Width_mean=("Mean95Width", "mean"),
            UncertaintyAbsErrorSpearman_mean=("UncertaintyAbsErrorSpearman", "mean"),
        )
        .sort_values("RMSE_mean")
        .reset_index(drop=True)
    )

    return folds, summary, preds


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

def savefig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()
    logging.info("Saved figure: %s", path)


def plot_sampling_layout(df: pd.DataFrame, out: Path) -> None:
    hole_mean = (
        df.groupby("HoleID", as_index=False)
        .agg(Easting_m=("Easting_m", "first"), Northing_m=("Northing_m", "first"), MeanRMR=("RMR", "mean"))
    )
    plt.figure(figsize=(9, 7))
    sc = plt.scatter(hole_mean["Easting_m"], hole_mean["Northing_m"], c=hole_mean["MeanRMR"], s=70)
    plt.colorbar(sc, label="Mean RMR per drill hole")
    plt.xlabel("Easting (m)")
    plt.ylabel("Northing (m)")
    plt.title("Synthetic drill-hole collar layout")
    plt.xlim(DOMAIN.x_min, DOMAIN.x_max)
    plt.ylim(DOMAIN.y_min, DOMAIN.y_max)
    plt.gca().set_aspect("equal", adjustable="box")
    savefig(out / "fig_01_sampling_layout.png")


def plot_3d_samples(df: pd.DataFrame, out: Path) -> None:
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(
        df["Easting_m"],
        df["Northing_m"],
        -df["Depth_m"],
        c=df["RMR"],
        s=16,
        alpha=0.85,
    )
    fig.colorbar(sc, ax=ax, shrink=0.72, label="RMR")
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.set_zlabel("Elevation relative to collar (m)")
    ax.set_title("3D synthetic drill-hole RMR samples")
    savefig(out / "fig_02_3d_drillhole_samples.png")


def plot_variogram(ok: OrdinaryKriging3D, out: Path) -> None:
    assert ok.empirical_ is not None and ok.params_ is not None
    empirical = ok.empirical_
    nugget, psill, range_ = ok.params_

    h = np.linspace(0.0, empirical["lag"].max() * 1.15, 250)
    fitted = spherical_variogram(h, nugget, psill, range_)

    plt.figure(figsize=(8.5, 6.2))
    plt.scatter(
        empirical["lag"],
        empirical["semivariance"],
        s=np.clip(empirical["pairs"].to_numpy() / 8.0, 25, 180),
        label="Experimental semivariogram",
    )
    plt.plot(h, fitted, linewidth=2.0, label="Fitted spherical model")
    plt.axvline(range_, linestyle="--", linewidth=1.3, label=f"Range = {range_:.2f} scaled units")
    plt.xlabel("Anisotropy-scaled separation distance")
    plt.ylabel("Semivariance")
    plt.title("Experimental and fitted RMR semivariogram")
    plt.legend()
    txt = f"Nugget={nugget:.2f}\nPartial sill={psill:.2f}\nTotal sill={nugget + psill:.2f}"
    plt.gca().text(0.98, 0.05, txt, transform=plt.gca().transAxes, ha="right", va="bottom")
    savefig(out / "fig_03_variogram.png")


def plot_cv_scatter(preds: pd.DataFrame, out: Path) -> None:
    models = ["Ordinary Kriging", "Gaussian Process", "Random Forest"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), sharex=True, sharey=True)
    lims = [max(RMR_MIN, preds["ObservedRMR"].min() - 4), min(RMR_MAX, preds["ObservedRMR"].max() + 4)]

    for ax, model in zip(axes, models):
        d = preds[preds["Model"] == model]
        ax.scatter(d["ObservedRMR"], d["PredictedRMR"], s=18, alpha=0.65)
        ax.plot(lims, lims, linestyle="--", linewidth=1.4)
        ax.set_title(model)
        ax.set_xlabel("Observed RMR")
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Spatial-CV predicted RMR")
    fig.suptitle("Out-of-fold predictions under spatially blocked validation", y=1.02)
    savefig(out / "fig_04_spatial_cv_scatter.png")


def plot_cv_metrics(summary: pd.DataFrame, out: Path) -> None:
    order = ["Ordinary Kriging", "Gaussian Process", "Random Forest"]
    s = summary.set_index("Model").loc[order].reset_index()

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.0))

    axes[0].bar(s["Model"], s["RMSE_mean"], yerr=s["RMSE_sd"], capsize=4)
    axes[0].set_ylabel("RMR points")
    axes[0].set_title("Spatial CV: RMSE")

    axes[1].bar(s["Model"], s["MAE_mean"], yerr=s["MAE_sd"], capsize=4)
    axes[1].set_ylabel("RMR points")
    axes[1].set_title("Spatial CV: MAE")

    axes[2].bar(s["Model"], s["Coverage95_mean"])
    axes[2].axhline(0.95, linestyle="--", linewidth=1.4, label="Nominal 95%")
    axes[2].set_ylim(0.0, 1.05)
    axes[2].set_ylabel("Fraction of observations")
    axes[2].set_title("95% interval coverage")
    axes[2].legend()

    for ax in axes:
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=0.2)

    fig.suptitle("Model performance and uncertainty validation", y=1.02)
    savefig(out / "fig_05_cv_metrics.png")


def _map_extent() -> Tuple[float, float, float, float]:
    return (DOMAIN.x_min, DOMAIN.x_max, DOMAIN.y_min, DOMAIN.y_max)


def _imshow_map(ax, grid: np.ndarray, title: str, vmin=None, vmax=None, cmap=None):
    im = ax.imshow(
        grid,
        origin="lower",
        extent=_map_extent(),
        aspect="equal",
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
    )
    ax.set_title(title)
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    return im


def plot_prediction_maps(
    map_arrays: Dict[str, np.ndarray],
    collars: pd.DataFrame,
    depth: float,
    out: Path,
) -> None:
    names = ["Synthetic truth", "Ordinary Kriging", "Gaussian Process", "Random Forest"]
    all_values = np.concatenate([map_arrays[n].ravel() for n in names])
    vmin, vmax = np.quantile(all_values, [0.01, 0.99])

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    for ax, name in zip(axes.ravel(), names):
        im = _imshow_map(ax, map_arrays[name], name, vmin=vmin, vmax=vmax)
        ax.scatter(collars["Easting_m"], collars["Northing_m"], s=11, facecolors="none", edgecolors="black", linewidths=0.55)
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.80, label="RMR")
    fig.suptitle(f"RMR prediction maps at depth = {depth:.1f} m", y=0.98)
    fig.subplots_adjust(top=0.92, right=0.90, wspace=0.18, hspace=0.20)
    plt.savefig(out / "fig_06_prediction_maps.png", dpi=220, bbox_inches="tight")
    plt.close()


def plot_uncertainty_maps(
    std_maps: Dict[str, np.ndarray],
    collars: pd.DataFrame,
    depth: float,
    out: Path,
) -> None:
    names = ["Ordinary Kriging", "Gaussian Process", "Random Forest"]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.3))

    for ax, name in zip(axes, names):
        title = name if name != "Random Forest" else "Random Forest (tree-spread proxy)"
        im = _imshow_map(ax, std_maps[name], title)
        ax.scatter(collars["Easting_m"], collars["Northing_m"], s=10, facecolors="none", edgecolors="black", linewidths=0.5)
        fig.colorbar(im, ax=ax, shrink=0.82, label="Prediction standard deviation / spread")

    fig.suptitle(f"Uncertainty maps at depth = {depth:.1f} m", y=1.02)
    savefig(out / "fig_07_uncertainty_maps.png")


def plot_absolute_error_maps(
    truth: np.ndarray,
    prediction_maps: Dict[str, np.ndarray],
    depth: float,
    out: Path,
) -> None:
    names = ["Ordinary Kriging", "Gaussian Process", "Random Forest"]
    errors = {name: np.abs(prediction_maps[name] - truth) for name in names}
    vmax = max(float(np.quantile(e, 0.98)) for e in errors.values())

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.3))
    for ax, name in zip(axes, names):
        im = _imshow_map(ax, errors[name], name, vmin=0.0, vmax=vmax)
        fig.colorbar(im, ax=ax, shrink=0.82, label="Absolute error (RMR points)")
    fig.suptitle(f"Absolute interpolation error against synthetic truth at depth = {depth:.1f} m", y=1.02)
    savefig(out / "fig_08_absolute_error_maps.png")


def plot_uncertainty_vs_error(preds: pd.DataFrame, out: Path) -> None:
    models = ["Ordinary Kriging", "Gaussian Process", "Random Forest"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))

    for ax, model in zip(axes, models):
        d = preds[preds["Model"] == model].copy()
        d["AbsError"] = np.abs(d["PredictedRMR"] - d["ObservedRMR"])
        rho = spearmanr(d["PredStd"], d["AbsError"], nan_policy="omit").statistic
        ax.scatter(d["PredStd"], d["AbsError"], s=18, alpha=0.6)
        ax.set_title(f"{model}\nSpearman ρ = {rho:.2f}")
        ax.set_xlabel("Predicted uncertainty / spread")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Absolute CV error (RMR points)")
    fig.suptitle("Does higher reported uncertainty coincide with larger errors?", y=1.02)
    savefig(out / "fig_09_uncertainty_vs_error.png")


# -----------------------------------------------------------------------------
# Final model fitting and mapping
# -----------------------------------------------------------------------------

def fit_final_models(df: pd.DataFrame, seed: int = 42):
    coords = transform_coordinates(dataframe_coordinates(df))
    y = df["RMR"].to_numpy(dtype=float)

    logging.info("Fitting final Ordinary Kriging model...")
    ok = OrdinaryKriging3D(n_lags=12).fit(coords, y)

    logging.info("Fitting final Gaussian Process model...")
    gpr = build_gpr(seed)
    gpr.fit(coords, y)

    logging.info("Fitting final Random Forest model...")
    rf = build_random_forest(seed)
    rf.fit(coords, y)

    return ok, gpr, rf


def make_depth_slice_predictions(
    df: pd.DataFrame,
    ok: OrdinaryKriging3D,
    gpr: GaussianProcessRegressor,
    rf: RandomForestRegressor,
    depth: float,
    grid_size: int = 80,
):
    xg = np.linspace(DOMAIN.x_min, DOMAIN.x_max, grid_size)
    yg = np.linspace(DOMAIN.y_min, DOMAIN.y_max, grid_size)
    xx, yy = np.meshgrid(xg, yg)
    dd = np.full_like(xx, float(depth))

    query_unscaled = np.column_stack([xx.ravel(), yy.ravel(), dd.ravel()])
    query = transform_coordinates(query_unscaled)

    logging.info("Predicting %d grid cells with Ordinary Kriging...", len(query))
    ok_mean, ok_std = ok.predict(query)

    logging.info("Predicting %d grid cells with Gaussian Process...", len(query))
    gpr_mean, gpr_std = gpr.predict(query, return_std=True)

    logging.info("Predicting %d grid cells with Random Forest...", len(query))
    rf_mean, rf_std = random_forest_predict_with_spread(rf, query)

    truth = true_rmr_function(xx, yy, dd)

    shape = xx.shape
    prediction_maps = {
        "Synthetic truth": truth,
        "Ordinary Kriging": ok_mean.reshape(shape),
        "Gaussian Process": gpr_mean.reshape(shape),
        "Random Forest": rf_mean.reshape(shape),
    }
    std_maps = {
        "Ordinary Kriging": ok_std.reshape(shape),
        "Gaussian Process": gpr_std.reshape(shape),
        "Random Forest": rf_std.reshape(shape),
    }

    slice_table = pd.DataFrame(
        {
            "Easting_m": xx.ravel(),
            "Northing_m": yy.ravel(),
            "Depth_m": dd.ravel(),
            "SyntheticTrueRMR": truth.ravel(),
            "Kriging_RMR": ok_mean,
            "Kriging_SD": ok_std,
            "GPR_RMR": gpr_mean,
            "GPR_SD": gpr_std,
            "RF_RMR": rf_mean,
            "RF_TreeSpread": rf_std,
        }
    )

    collars = (
        df.groupby("HoleID", as_index=False)
        .agg(Easting_m=("Easting_m", "first"), Northing_m=("Northing_m", "first"))
    )

    return prediction_maps, std_maps, slice_table, collars


# -----------------------------------------------------------------------------
# Main pipeline
# -----------------------------------------------------------------------------

def run_pipeline(output_dir: str, seed: int, slice_depth: float, grid_size: int) -> None:
    setup_logging()
    out = ensure_output_dir(output_dir)

    if not (DOMAIN.depth_min <= slice_depth <= DOMAIN.depth_max):
        raise ValueError(f"slice_depth must be between {DOMAIN.depth_min} and {DOMAIN.depth_max} m")
    if grid_size < 25:
        raise ValueError("grid_size should be at least 25 for meaningful maps.")

    logging.info("STEP 1/8 - Generating synthetic drill-hole data")
    clean_truth_source = generate_synthetic_drillholes(seed=seed)
    raw = inject_data_quality_issues(clean_truth_source, seed=seed)
    raw.to_csv(out / "01_raw_synthetic_drillholes.csv", index=False)

    logging.info("STEP 2/8 - ETL and data-quality control")
    clean, audit = etl_clean_rmr(raw)
    clean.to_csv(out / "02_clean_drillholes.csv", index=False)
    with open(out / "02_etl_audit.json", "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)

    logging.info("ETL audit: %s", audit)
    logging.info(
        "Clean RMR summary: n=%d, holes=%d, mean=%.2f, sd=%.2f, min=%.2f, max=%.2f",
        len(clean),
        clean["HoleID"].nunique(),
        clean["RMR"].mean(),
        clean["RMR"].std(),
        clean["RMR"].min(),
        clean["RMR"].max(),
    )

    logging.info("STEP 3/8 - EDA figures")
    plot_sampling_layout(clean, out)
    plot_3d_samples(clean, out)

    logging.info("STEP 4/8 - Spatially blocked cross-validation")
    fold_metrics, summary, cv_predictions = spatial_cross_validate(clean, seed=seed)
    fold_metrics.to_csv(out / "03_cv_fold_metrics.csv", index=False)
    summary.to_csv(out / "04_cv_summary.csv", index=False)
    cv_predictions.to_csv(out / "05_cv_predictions.csv", index=False)
    print("\n=== SPATIALLY BLOCKED CROSS-VALIDATION SUMMARY ===")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:0.3f}"))

    logging.info("STEP 5/8 - CV diagnostic figures")
    plot_cv_scatter(cv_predictions, out)
    plot_cv_metrics(summary, out)
    plot_uncertainty_vs_error(cv_predictions, out)

    logging.info("STEP 6/8 - Fit final models on all cleaned samples")
    ok, gpr, rf = fit_final_models(clean, seed=seed)
    plot_variogram(ok, out)

    nugget, partial_sill, variogram_range = ok.params_
    model_info = {
        "ordinary_kriging": {
            "variogram": "spherical",
            "nugget": nugget,
            "partial_sill": partial_sill,
            "total_sill": nugget + partial_sill,
            "range_scaled_units": variogram_range,
            "anisotropy_scaling_m": {"Lx": DOMAIN.lx, "Ly": DOMAIN.ly, "Lz": DOMAIN.lz},
        },
        "gpr_optimized_kernel": str(gpr.kernel_),
        "random_forest": {
            "n_estimators": rf.n_estimators,
            "min_samples_leaf": rf.min_samples_leaf,
            "uncertainty_note": "Tree-to-tree standard deviation; heuristic ensemble-spread proxy, not formal spatial predictive variance.",
        },
    }
    with open(out / "06_fitted_model_info.json", "w", encoding="utf-8") as f:
        json.dump(model_info, f, indent=2)

    logging.info("STEP 7/8 - Generate depth-slice predictions and uncertainty maps")
    prediction_maps, std_maps, slice_table, collars = make_depth_slice_predictions(
        clean,
        ok,
        gpr,
        rf,
        depth=slice_depth,
        grid_size=grid_size,
    )
    slice_table.to_csv(out / f"07_slice_predictions_depth_{slice_depth:.1f}m.csv", index=False)

    plot_prediction_maps(prediction_maps, collars, slice_depth, out)
    plot_uncertainty_maps(std_maps, collars, slice_depth, out)
    plot_absolute_error_maps(prediction_maps["Synthetic truth"], prediction_maps, slice_depth, out)

    logging.info("STEP 8/8 - Save map-level synthetic truth diagnostics")
    map_metrics = []
    truth = prediction_maps["Synthetic truth"].ravel()
    for name in ["Ordinary Kriging", "Gaussian Process", "Random Forest"]:
        pred = prediction_maps[name].ravel()
        std = std_maps[name].ravel()
        map_metrics.append({"Model": name, **metric_row(truth, pred, std)})
    map_metrics_df = pd.DataFrame(map_metrics).sort_values("RMSE")
    map_metrics_df.to_csv(out / "08_depth_slice_truth_metrics.csv", index=False)

    print("\n=== DEPTH-SLICE METRICS AGAINST KNOWN SYNTHETIC TRUTH ===")
    print(map_metrics_df.to_string(index=False, float_format=lambda x: f"{x:0.3f}"))

    logging.info("Pipeline complete. Outputs written to: %s", out.resolve())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare Kriging, GPR, and Random Forest for synthetic drill-hole RMR interpolation.")
    parser.add_argument("--output-dir", default="rmr_spatial_outputs", help="Directory for CSVs, JSON, and figures.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--slice-depth", type=float, default=100.0, help="Depth (m) for 2D prediction/uncertainty maps.")
    parser.add_argument("--grid-size", type=int, default=80, help="Number of cells along each map axis.")
    args = parser.parse_args()

    run_pipeline(
        output_dir=args.output_dir,
        seed=args.seed,
        slice_depth=args.slice_depth,
        grid_size=args.grid_size,
    )
