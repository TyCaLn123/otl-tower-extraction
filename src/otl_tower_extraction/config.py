from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, List
import yaml


@dataclass
class StageConfig:
    """Switches for enabling or disabling major pipeline stages."""
    preprocess: bool = True
    localization: bool = True
    precise_extraction: bool = True


@dataclass
class ProgressConfig:
    """Progress display options."""
    enabled: bool = True
    stage_messages: bool = True
    open3d_progress: bool = True


@dataclass
class CoordinateConfig:
    """
    Coordinate scaling options.

    The recommended public-release convention is:
    - input TXT coordinates are in meters and rounded to 0.1 m;
    - internal computation uses integer decimeter coordinates;
    - output TXT coordinates are converted back to meters.

    If `integer_internal` is true and `scale` is 10, an input coordinate 12.3 m
    is internally represented as integer 123.
    """
    integer_internal: bool = True
    scale: int = 10
    input_unit: str = "m"
    output_unit: str = "m"


@dataclass
class PreprocessConfig:
    voxel_size: float = 0.2
    sor_neighbors: int = 20
    sor_std_ratio: float = 2.0
    grid_size: float = 1.0
    near_ground_height: float = 15


@dataclass
class LocalizationConfig:
    dbscan_voxel_size:  float = 1.0
    dbscan_eps: float = 0.8
    dbscan_min_points: int = 200
    min_cluster_height: float = 15.0
    bbox_expand_xy: float = 2.0


@dataclass
class CentralRegionConfig:
    slice_height: float = 1.0
    slice_window_height: float = 2.0
    angle_range_threshold_deg: float = 5.0
    rectangle_diff_threshold: float = 1.0
    angle_std_factor: float = 1.0

    # Ablation option. If true, the least-disturbed central region is not
    # selected. Instead, all slice descriptors from the entire candidate point
    # cloud are used to estimate pose and structural references.
    use_whole_candidate: bool = False

    # Adaptive central-region validation.
    # Only slice_height is adjusted automatically. The window height and other
    # central-region thresholds keep their default values.
    adaptive_slope_check: bool = True
    slope_abs_diff_threshold: float = 0.20
    adaptive_slice_heights: List[float] = field(
        default_factory=lambda: [0.1, 0.2, 0.5, 2.0]
    )

@dataclass
class SideViewConfig:
    enabled: bool = True
    vertical_step: float = 0.1
    tolerance: float = 0.2


@dataclass
class FrontViewConfig:
    enabled: bool = True
    vertical_step: float = 0.1
    tolerance: float = 0.5
    raster_margin: int = 10
    jump_threshold: float = 2.0


@dataclass
class BaseFilterConfig:
    enabled: bool = True
    # Ablation option. If true, geometric base-model reconstruction is replaced
    # by a simple height-threshold filter.
    use_height_threshold_only: bool = False
    height_threshold: float = 0.5
    search_step: float = 0.1
    axis_contact_threshold: float = 0.2
    center_contact_gap: float = 0.5
    transition_cluster_gap: float = 1.0
    base_plane_thickness: float = 1.0
    base_top_score_lambda: float = 1.0
    lower_margin: float = 0.5
    fallback_base_height: float = 5.0
    distance_tolerance: float = 1.0


@dataclass
class PipelineConfig:
    input_path: str
    output_dir: str = "outputs"
    output_prefix: str = "tower"
    label_column: Optional[int] = 3
    save_point_clouds: bool = True
    continue_on_candidate_error: bool = True
    coordinates: CoordinateConfig = field(default_factory=CoordinateConfig)
    stages: StageConfig = field(default_factory=StageConfig)
    progress: ProgressConfig = field(default_factory=ProgressConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    localization: LocalizationConfig = field(default_factory=LocalizationConfig)
    central_region: CentralRegionConfig = field(default_factory=CentralRegionConfig)
    side_view: SideViewConfig = field(default_factory=SideViewConfig)
    front_view: FrontViewConfig = field(default_factory=FrontViewConfig)
    base_filter: BaseFilterConfig = field(default_factory=BaseFilterConfig)


def _update_dataclass(obj: Any, values: Dict[str, Any]) -> Any:
    for key, value in values.items():
        if key == "base_config":
            continue
        if not hasattr(obj, key):
            continue
        current = getattr(obj, key)
        if hasattr(current, "__dataclass_fields__") and isinstance(value, dict):
            _update_dataclass(current, value)
        else:
            setattr(obj, key, value)
    return obj


def load_config(path: str | Path) -> PipelineConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if "input_path" not in raw:
        raise ValueError("Configuration must contain `input_path`.")
    cfg = PipelineConfig(input_path=raw["input_path"])
    return _update_dataclass(cfg, raw)


def _scale_attr(obj: Any, name: str, scale: float, min_int: bool = False) -> None:
    value = getattr(obj, name)
    scaled = value * scale
    if min_int:
        scaled = max(1, int(round(scaled)))
    setattr(obj, name, scaled)


def make_internal_config(cfg: PipelineConfig) -> PipelineConfig:
    """
    Return a copy of the configuration whose geometric parameters are expressed
    in the same unit as the internal point coordinates.

    When integer_internal is enabled, input point coordinates are multiplied by
    `coordinates.scale`. Therefore, all distance-related parameters are also
    multiplied by the same scale before being used by the algorithm.
    """
    internal = deepcopy(cfg)
    if not cfg.coordinates.integer_internal:
        return internal

    s = float(cfg.coordinates.scale)

    # Preprocessing distances.
    _scale_attr(internal.preprocess, "voxel_size", s, min_int=False)
    _scale_attr(internal.preprocess, "grid_size", s, min_int=False)
    _scale_attr(internal.preprocess, "near_ground_height", s, min_int=False)

    # Localization distances.
    _scale_attr(internal.localization, "dbscan_voxel_size", s, min_int=False)
    _scale_attr(internal.localization, "dbscan_eps", s, min_int=False)
    _scale_attr(internal.localization, "min_cluster_height", s, min_int=False)
    _scale_attr(internal.localization, "bbox_expand_xy", s, min_int=False)

    # Central-region geometric lengths.
    _scale_attr(internal.central_region, "slice_height", s, min_int=False)
    _scale_attr(internal.central_region, "slice_window_height", s, min_int=False)
    _scale_attr(internal.central_region, "rectangle_diff_threshold", s, min_int=False)

    # View refinement parameters.
    _scale_attr(internal.side_view, "vertical_step", s, min_int=False)
    _scale_attr(internal.side_view, "tolerance", s, min_int=False)
    _scale_attr(internal.front_view, "vertical_step", s, min_int=False)
    _scale_attr(internal.front_view, "tolerance", s, min_int=False)
    _scale_attr(internal.front_view, "jump_threshold", s, min_int=False)
    # raster_margin is measured in raster cells, so it is intentionally not scaled.

    # Base filtering distances.
    _scale_attr(internal.base_filter, "height_threshold", s, min_int=False)
    _scale_attr(internal.base_filter, "search_step", s, min_int=False)
    _scale_attr(internal.base_filter, "axis_contact_threshold", s, min_int=False)
    _scale_attr(internal.base_filter, "center_contact_gap", s, min_int=False)
    _scale_attr(internal.base_filter, "transition_cluster_gap", s, min_int=False)
    _scale_attr(internal.base_filter, "base_plane_thickness", s, min_int=False)
    _scale_attr(internal.base_filter, "lower_margin", s, min_int=False)
    _scale_attr(internal.base_filter, "fallback_base_height", s, min_int=False)
    _scale_attr(internal.base_filter, "distance_tolerance", s, min_int=False)

    if hasattr(internal.central_region, "adaptive_slice_heights"):
        internal.central_region.adaptive_slice_heights = [
            float(v) * s for v in internal.central_region.adaptive_slice_heights
        ]

    return internal
