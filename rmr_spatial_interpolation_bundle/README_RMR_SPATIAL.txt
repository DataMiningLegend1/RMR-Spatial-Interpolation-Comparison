RMR Spatial Interpolation Comparison
====================================

Files
-----
1. rmr_spatial_interpolation_comparison.py
   Full end-to-end Python workflow.
2. requirements_rmr_spatial.txt
   Required Python packages.
3. rmr_test_outputs/
   Verified example outputs generated with seed=42, depth=100 m, grid=35.

Install
-------
python -m pip install -r requirements_rmr_spatial.txt

Run
---
python rmr_spatial_interpolation_comparison.py

Example
-------
python rmr_spatial_interpolation_comparison.py --output-dir rmr_spatial_outputs --seed 42 --slice-depth 100 --grid-size 80

What the workflow does
----------------------
- Generates synthetic vertical drill holes and noisy RMR observations.
- Injects missing values, duplicates, malformed numeric fields, and impossible RMR values.
- Performs ETL, physical-domain checks, and an audit trail.
- Uses anisotropy-scaled 3D coordinates (Easting, Northing, Depth).
- Computes an experimental semivariogram and fits a spherical variogram.
- Implements Ordinary Kriging directly from the kriging linear system.
- Fits Gaussian Process Regression with a Matern kernel.
- Fits Random Forest regression.
- Uses XY spatial-block cross-validation rather than random row splitting.
- Reports RMSE, MAE, R2, bias, 95% interval coverage, interval width, and correlation between predicted uncertainty and absolute error.
- Produces prediction, uncertainty, and absolute-error maps on a selected depth slice.

Uncertainty interpretation
--------------------------
Ordinary Kriging: sqrt(kriging variance).
GPR: posterior predictive standard deviation from the fitted GP.
Random Forest: tree-to-tree standard deviation only. This is a heuristic ensemble-spread proxy and should not be described as equivalent to kriging variance/GPR posterior variance.

Using real mine data
--------------------
Replace generate_synthetic_drillholes() + inject_data_quality_issues() with pd.read_csv(...) or your database extract. Keep columns:
HoleID, Easting_m, Northing_m, Depth_m, RMR.

Then remove SyntheticTrueRMR-based map diagnostics because real unsampled truth is unknown. Retain spatial cross-validation for defensible model comparison.
