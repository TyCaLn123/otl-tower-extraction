# Shape Prior-Guided Coarse-to-Fine Extraction of Overhead Transmission Line Towers from UAV LiDAR Point Clouds

## About

This repository is the public release page for a framework that extracts overhead
transmission line towers from large-scale UAV LiDAR point clouds.

The proposed method follows a shape prior-guided coarse-to-fine strategy. It is
designed for scene point clouds containing tens to hundreds of millions of
points and processes each tower in approximately 100 to 300 seconds, depending
on scene density.

## Publication

The article has been published in *Remote Sensing*:

> Tong, C., Shen, Y., Zhang, K., and Wei, H. (2026). Shape
> Prior-Guided Coarse-to-Fine Extraction of Overhead Transmission Line Towers
> from UAV LiDAR Point Clouds. *Remote Sensing*, 18(13), 2082.
> https://doi.org/10.3390/rs18132082

## Repository Contents

This public release includes:

- source code for the complete extraction pipeline;
- configuration files for the proposed method;
- environment and dependency requirements;
- command-line examples for running the pipeline;
- documentation for input data preparation and output interpretation;
- scripts for evaluation and selected experiments.

## Dataset Folder Preparation

The original UAV LiDAR datasets used in the article are not included in this
repository. The example configurations use `data/` as the dataset folder. To run
the released code, prepare your own `data/` folder under the repository root and
place each OTL corridor scene in a separate line folder:

```text
data/
  Line1/
    Line1.txt
  Line2/
    Line2.txt
  Line3/
    Line3.txt
  ...
```

Each scene file must be a plain space-separated TXT file without a header. The
default loader expects at least four columns:

```text
x y z label
```

- `x`, `y`, and `z` are point coordinates in meters.
- `label` is the binary ground-truth label, where `1` denotes tower points and
  `0` denotes non-tower/background points.
- The default configuration uses `label_column: 3`, meaning that the fourth
  column is read as the label.

Example:

```text
431256.7 3120451.2 48.6 0
431257.1 3120451.4 72.3 1
431257.4 3120451.8 73.0 1
```

If your original point clouds are stored as LAS, LAZ, PLY, or another format,
convert them to the TXT format above before running the scripts. For unlabeled
inference-only use, the current loader still requires a fourth column, so add a
placeholder label column and ignore the metric CSV files.

After preparing the files, update `input_path` in `configs/default.yaml`, for
example:

```yaml
input_path: data/Line1/Line1.txt
```

For batch ablation or sensitivity experiments, list all prepared scene files in
the `input_paths` field of `configs/ablation.yaml` or
`configs/sensitivity.yaml`.

If you prefer another folder name such as `dataset/`, keep the same TXT file
format and update all corresponding `input_path` or `input_paths` entries.

## Data Availability

The original datasets used in this study were collected in collaboration with
an industrial partner and cannot be distributed publicly without permission.
Information about any releasable sample data will be added to this repository
if such data can be distributed later.

## Citation

If you use this repository or find the work helpful, please cite:

```bibtex
@Article{rs18132082,
  AUTHOR = {Tong, Chaoliu and Shen, Yu and Zhang, Kanjian and Wei, Haikun},
  TITLE = {Shape Prior-Guided Coarse-to-Fine Extraction of Overhead Transmission Line Towers from UAV LiDAR Point Clouds},
  JOURNAL = {Remote Sensing},
  VOLUME = {18},
  YEAR = {2026},
  NUMBER = {13},
  ARTICLE-NUMBER = {2082},
  URL = {https://www.mdpi.com/2072-4292/18/13/2082},
  ISSN = {2072-4292},
  DOI = {10.3390/rs18132082}
}
```
