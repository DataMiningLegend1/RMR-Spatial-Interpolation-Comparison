# RMR Spatial Interpolation Comparison

A reproducible Python workflow for comparing **Ordinary Kriging**, **Gaussian Process Regression (GPR)**, and **Random Forest (RF)** for three-dimensional Rock Mass Rating (RMR) interpolation and spatial uncertainty quantification from synthetic drill-hole data.

## Project Overview

This project demonstrates a complete spatial-modeling pipeline:

**Synthetic drill holes → ETL/QC → 3D spatial transformation → variogram analysis → spatial cross-validation → Kriging/GPR/RF → uncertainty validation → prediction maps → uncertainty maps → error analysis**

The workflow was designed to resemble a realistic mining geostatistics problem rather than a simple machine-learning benchmark. The synthetic RMR field includes spatial variability, depth effects, a competent rock domain, a weak/fault-like zone, and a deeper weak lens.

## Objectives

The project aims to:

- Generate and clean synthetic drill-hole RMR data.
- Demonstrate a transparent ETL and data-quality workflow.
- Model RMR as a three-dimensional function of Easting, Northing, and Depth.
- Compare Ordinary Kriging, Gaussian Process Regression, and Random Forest.
- Evaluate model performance using spatially blocked cross-validation.
- Quantify and visualize spatial uncertainty.
- Produce traceable CSV, JSON, and graphical outputs suitable for research, teaching, and technical demonstrations.

## Methods

### 1. ETL and Data Quality Control

The workflow deliberately introduces data-quality issues before modeling, including:

- Missing RMR values
- Missing coordinates
- Duplicate records
- Physically invalid RMR values
- Numeric depth values stored as strings

The ETL stage performs:

- Numeric coercion
- Required-field validation
- Duplicate removal
- Missing-value removal
- Spatial-domain checks
- RMR physical-bound validation
- Sorting and audit logging

Example verified ETL results:

| Metric | Value |
|---|---:|
| Raw rows | 324 |
| Duplicates removed | 4 |
| Missing required removed | 5 |
| Physically invalid removed | 3 |
| Final valid samples | 312 |
| Drill holes retained | 32 |

### 2. Three-Dimensional Spatial Representation

Each sample is represented by:

- `Easting_m`
- `Northing_m`
- `Depth_m`
- `RMR`

The models therefore estimate:

**RMR = f(X, Y, Z)**

rather than interpolating only a two-dimensional collar surface.

Anisotropy scaling is also included so that horizontal and vertical distances are not automatically treated as equivalent.

### 3. Ordinary Kriging

The script calculates an experimental semivariogram and fits a spherical variogram model.

The Ordinary Kriging system is solved directly in Python rather than relying on PyKrige. This makes the implementation useful for:

- Research demonstrations
- Coursework
- Methodology explanation
- Technical review
- Transparent geostatistical experimentation

The model returns both predicted RMR values and kriging uncertainty.

### 4. Gaussian Process Regression

Gaussian Process Regression is implemented using `GaussianProcessRegressor` with a Matérn covariance kernel and white-noise component.

The model produces:

- Posterior mean RMR predictions
- Posterior standard deviation

This enables direct probabilistic uncertainty mapping.

### 5. Random Forest

Random Forest is trained using the same transformed spatial coordinates.

The implementation uses:

- 400 trees
- Bootstrap sampling
- `min_samples_leaf=2`

Random Forest does not inherently provide geostatistical uncertainty. Therefore, the project uses the standard deviation across tree predictions as a **tree-spread uncertainty proxy**.

This should not be interpreted as equivalent to kriging variance or Gaussian-process posterior variance.

## Spatial Cross-Validation

The project uses spatially blocked cross-validation rather than randomly splitting individual drill-hole intervals.

`GroupKFold` is used so that entire geographical regions are held out during validation.

This reduces spatial leakage and provides a more realistic test:

> Can the model estimate RMR in a spatial region where no training samples are available?

## Validation Metrics

The workflow calculates:

- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- R²
- Prediction bias
- 95% interval coverage
- Mean 95% interval width
- Spearman correlation between predicted uncertainty and absolute error

The uncertainty-error correlation is particularly useful because a meaningful uncertainty model should assign higher uncertainty to locations where prediction errors are more likely to be large.

## Example Spatial Cross-Validation Results

| Model | RMSE | MAE | Mean R² | 95% Coverage |
|---|---:|---:|---:|---:|
| Gaussian Process | 8.07 | 6.88 | -0.456 | 87.6% |
| Ordinary Kriging | 8.65 | 7.34 | -0.618 | 89.2% |
| Random Forest | 9.75 | 8.01 | -1.120 | 62.5% |

These values describe this deliberately difficult synthetic spatial-block experiment and should **not** be interpreted as universal rankings of the algorithms.

## Example Depth-Slice Results

For the generated synthetic truth at the modeled depth slice:

| Model | RMSE | MAE | R² |
|---|---:|---:|---:|
| Gaussian Process | 3.65 | 2.63 | 0.789 |
| Ordinary Kriging | 4.30 | 3.18 | 0.706 |
| Random Forest | 5.31 | 3.78 | 0.551 |

Again, these results apply only to the synthetic experiment included in this repository.

## Generated Figures

The workflow automatically generates:

1. `fig_01_sampling_layout.png`
2. `fig_02_3d_drillhole_samples.png`
3. `fig_03_variogram.png`
4. `fig_04_spatial_cv_scatter.png`
5. `fig_05_cv_metrics.png`
6. `fig_06_prediction_maps.png`
7. `fig_07_uncertainty_maps.png`
8. `fig_08_absolute_error_maps.png`
9. `fig_09_uncertainty_vs_error.png`

The prediction comparison includes:

- Synthetic truth
- Ordinary Kriging
- Gaussian Process Regression
- Random Forest

The uncertainty comparison includes:

- Kriging standard deviation
- GPR posterior standard deviation
- Random Forest tree-spread uncertainty proxy

## Output Files

The workflow saves research outputs including:

- `01_raw_synthetic_drillholes.csv`
- `02_clean_drillholes.csv`
- `02_etl_audit.json`
- `03_cv_fold_metrics.csv`
- `04_cv_summary.csv`
- `05_cv_predictions.csv`
- `06_fitted_model_info.json`
- `07_slice_predictions_depth_100.0m.csv`
- `08_depth_slice_truth_metrics.csv`

These outputs can be reused in:

- Power BI
- Excel
- ArcGIS
- QGIS
- Surpac
- Vulcan
- Leapfrog
- Digital-twin dashboards

## Repository Structure

```text
RMR-Spatial-Interpolation-Comparison/
│
├── README.md
├── rmr_spatial_interpolation_bundle.zip
│
├── rmr_spatial_interpolation_bundle/
│   ├── README_RMR_SPATIAL.txt
│   ├── requirements_rmr_spatial.txt
│   ├── rmr_spatial_interpolation_comparison.py
│   └── rmr_test_outputs/
│       ├── CSV / JSON outputs
│       └── Generated figures
│
└── rmr_spatial_outputs/
    ├── CSV / JSON outputs
    └── Generated figures
```

## Installation

Clone the repository:

```bash
git clone https://github.com/DataMiningLegend1/RMR-Spatial-Interpolation-Comparison.git
cd RMR-Spatial-Interpolation-Comparison
```

Install dependencies:

```bash
python -m pip install -r rmr_spatial_interpolation_bundle/requirements_rmr_spatial.txt
```

## Running the Project

Run with the default configuration:

```bash
python rmr_spatial_interpolation_bundle/rmr_spatial_interpolation_comparison.py
```

Or specify output directory, random seed, slice depth, and grid size:

```bash
python rmr_spatial_interpolation_bundle/rmr_spatial_interpolation_comparison.py     --output-dir rmr_results     --seed 42     --slice-depth 100     --grid-size 80
```

Example alternative depth slices:

```bash
--slice-depth 50
--slice-depth 100
--slice-depth 150
```

## Using Real Drill-Hole Data

To adapt the workflow to real RMR data, replace the synthetic-data generator with a CSV import such as:

```python
df = pd.read_csv("rmr_drillholes.csv")
```

Expected fields include:

```text
HoleID
Easting_m
Northing_m
Depth_m
RMR
```

For deviated drill holes, use desurveyed XYZ coordinates for each RMR interval centroid rather than relying only on collar coordinates and measured depth.

With real data, the synthetic-truth error-map section should be removed because the true RMR at unsampled locations is unknown. Spatial cross-validation then becomes the main defensible measure of predictive performance.

## Research Framing

A suitable research framing for this work is:

**Comparative Evaluation of Geostatistical and Machine-Learning Methods for Three-Dimensional RMR Interpolation and Spatial Uncertainty Quantification from Drill-Hole Data**

## Key Technical Takeaways

- Kriging and Gaussian Process Regression both model spatial dependence and naturally provide uncertainty estimates.
- Random Forest can capture nonlinear spatial patterns but does not inherently provide geostatistical uncertainty.
- Spatially blocked validation is more defensible than random train/test splitting for drill-hole interpolation.
- Uncertainty maps should be evaluated quantitatively, not only visually.
- Synthetic truth provides a controlled environment for testing interpolation and uncertainty behavior before moving to real mine data.

## Author

**Davie Mdumuka**  
Mining Engineer | Data Analyst  
Python | SQL | Power BI | Machine Learning | Geostatistics

GitHub: `DataMiningLegend1`

## License

Add a license appropriate to your intended use before redistributing or incorporating external code or data.
