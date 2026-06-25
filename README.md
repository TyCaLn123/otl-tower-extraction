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
