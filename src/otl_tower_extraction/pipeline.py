from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import traceback
import numpy as np
import pandas as pd
from tqdm import tqdm

from .config import PipelineConfig, make_internal_config
from .data import load_point_cloud, save_point_cloud
from .preprocessing import voxel_downsample, statistical_outlier_removal
from .localization import localize_candidate_towers, CandidateTower, grid_near_ground_suppression
from .precise_extraction import TowerPreciseExtractor
from .metrics import binary_metrics
from .progress import ProgressReporter


def _nan_candidate_metrics(prefix: str = "candidate_") -> Dict[str, float]:
    return {
        f"{prefix}TP": float("nan"),
        f"{prefix}FP": float("nan"),
        f"{prefix}FN": float("nan"),
        f"{prefix}TN": float("nan"),
        f"{prefix}Precision": float("nan"),
        f"{prefix}Recall": float("nan"),
        f"{prefix}F1": float("nan"),
        f"{prefix}IoU": float("nan"),
        f"{prefix}OA": float("nan"),
    }


class TowerExtractionPipeline:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.icfg = make_internal_config(cfg)
        self.output_dir = Path(cfg.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.input_path = Path(cfg.input_path)
        self.input_stem = self.input_path.stem
        self.file_prefix = self._build_file_prefix()

        self.progress = ProgressReporter(
            enabled=cfg.progress.enabled,
            stage_messages=cfg.progress.stage_messages,
        )

    def _build_file_prefix(self) -> str:
        raw_prefix = self.cfg.output_prefix
        if raw_prefix is None:
            return self.input_stem

        raw_prefix = str(raw_prefix).strip()
        if raw_prefix == "" or raw_prefix.lower() == "auto":
            return self.input_stem

        if raw_prefix.startswith(self.input_stem):
            return raw_prefix

        return f"{self.input_stem}_{raw_prefix}"

    def _output_path(self, suffix: str) -> Path:
        return self.output_dir / f"{self.file_prefix}_{suffix}"

    def _write_candidate_summary(self, rows: list[dict], summary_path: Path) -> None:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(summary_path, index=False)

    def run(self) -> Dict[str, Any]:
        with self.progress.stage("Load input TXT"):
            data = load_point_cloud(
                self.cfg.input_path,
                self.cfg.label_column,
                integer_internal=self.cfg.coordinates.integer_internal,
                scale=self.cfg.coordinates.scale,
            )
            raw_points = data.points
            raw_labels = data.labels
            self.progress.log(
                f"  - input file: {self.input_path.name}\n"
                f"  - output prefix: {self.file_prefix}\n"
                f"  - input points: {len(raw_points):,}; "
                f"dtype={raw_points.dtype}; integer_internal={self.cfg.coordinates.integer_internal}; "
                f"scale={self.cfg.coordinates.scale}"
            )

        scene, scene_labels = self._run_preprocess(raw_points, raw_labels)
        scene_indices = np.arange(len(scene), dtype=np.int64)

        candidates = self._run_localization(scene, scene_labels, scene_indices)

        extractor = None
        if self.cfg.stages.precise_extraction:
            extractor = TowerPreciseExtractor(
                self.icfg.central_region,
                self.icfg.side_view,
                self.icfg.front_view,
                self.icfg.base_filter,
                show_progress=self.cfg.progress.enabled,
            )

        rows = []
        scene_pred = np.zeros(len(scene), dtype=np.int8)
        valid_scene_pred_mask = np.zeros(len(scene), dtype=bool)

        # Scene-level visualization labels after preprocessing.
        # 0: background/non-selected points; 1,2,...: candidate/tower instance ids.
        localization_scene_labels = np.zeros(len(scene), dtype=np.int32)
        precise_scene_labels = np.zeros(len(scene), dtype=np.int32)

        all_tower_points, all_tower_labels = [], []
        all_localized_points, all_localized_labels = [], []

        summary_path = self._output_path("summary.csv")
        scene_metrics_path = self._output_path("scene_metrics.csv")
        partial_line = False
        failed_candidates = 0
        successful_candidates = 0
        first_error_type = ""
        first_error_message = ""
        first_error_traceback = ""

        candidate_iter = enumerate(candidates)
        if self.cfg.progress.enabled:
            candidate_iter = tqdm(
                candidate_iter,
                total=len(candidates),
                desc="Candidate precise extraction",
                unit="candidate",
            )

        with self.progress.stage("Candidate processing and output"):
            for i, cand in candidate_iter:
                localized_path = self._output_path(f"candidate_{i:03d}_localized.txt")
                tower_path = self._output_path(f"candidate_{i:03d}_tower.txt")
                non_path = self._output_path(f"candidate_{i:03d}_non_tower.txt")

                # Save the candidate tower localization result before precise extraction.
                # This file corresponds to the recovered candidate region produced by
                # the localization stage and is saved even if later precise extraction fails.
                if self.cfg.save_point_clouds:
                    self._save_txt(localized_path, cand.points, cand.labels)

                # Mark the localization result in the preprocessed scene.
                # Positive labels are candidate instance ids: 1, 2, ...
                candidate_instance_id = i + 1
                localization_scene_labels[cand.source_indices] = candidate_instance_id

                if self.cfg.save_point_clouds and len(cand.points) > 0:
                    all_localized_points.append(cand.points)
                    if cand.labels is None:
                        all_localized_labels.append(np.ones(len(cand.points), dtype=np.int8))
                    else:
                        all_localized_labels.append(cand.labels)

                row = {
                    "input_file": self.input_path.name,
                    "input_stem": self.input_stem,
                    "output_prefix": self.file_prefix,
                    "candidate_id": i,
                    "cluster_id": cand.cluster_id,
                    "candidate_success": True,
                    "candidate_error_type": "",
                    "candidate_error_message": "",
                    "candidate_error_traceback": "",
                    "candidate_points": len(cand.points),
                    "localized_path": str(localized_path),
                    "tower_points": float("nan"),
                    "non_tower_points": float("nan"),
                    "tower_path": str(tower_path),
                    "non_tower_path": str(non_path),
                    "preprocess_enabled": self.cfg.stages.preprocess,
                    "localization_enabled": self.cfg.stages.localization,
                    "precise_extraction_enabled": self.cfg.stages.precise_extraction,
                }

                try:
                    if self.cfg.stages.precise_extraction:
                        result = extractor.extract(cand.points, cand.labels)
                        tower_points = result.tower_points
                        non_tower_points = result.non_tower_points
                        tower_labels = result.tower_labels
                        non_tower_labels = result.non_tower_labels
                        candidate_mask = result.mask
                    else:
                        candidate_mask = np.ones(len(cand.points), dtype=bool)
                        tower_points = cand.points
                        tower_labels = cand.labels
                        non_tower_points = np.empty((0, 3), dtype=cand.points.dtype)
                        non_tower_labels = np.empty((0,), dtype=np.int8) if cand.labels is not None else None

                    if self.cfg.save_point_clouds:
                        self._save_txt(tower_path, tower_points, tower_labels)
                        self._save_txt(non_path, non_tower_points, non_tower_labels)

                    # Only successful candidates contribute to scene prediction.
                    selected_scene_indices = cand.source_indices[candidate_mask]
                    scene_pred[selected_scene_indices] = 1
                    valid_scene_pred_mask[cand.source_indices] = True

                    # Mark the precise extraction result in the preprocessed scene.
                    # Positive labels are tower instance ids: 1, 2, ...
                    precise_scene_labels[selected_scene_indices] = candidate_instance_id

                    if self.cfg.save_point_clouds and len(tower_points) > 0:
                        all_tower_points.append(tower_points)
                        if tower_labels is None:
                            all_tower_labels.append(np.ones(len(tower_points), dtype=np.int8))
                        else:
                            all_tower_labels.append(tower_labels)

                    row["tower_points"] = len(tower_points)
                    row["non_tower_points"] = len(non_tower_points)

                    if cand.labels is not None:
                        candidate_metrics = binary_metrics(cand.labels.astype(int), candidate_mask.astype(int))
                        row.update({f"candidate_{k}": v for k, v in candidate_metrics.items()})

                    successful_candidates += 1

                except Exception as exc:
                    failed_candidates += 1
                    partial_line = True

                    err_type = type(exc).__name__
                    err_msg = str(exc)
                    err_tb = traceback.format_exc()

                    if not first_error_type:
                        first_error_type = err_type
                        first_error_message = err_msg
                        first_error_traceback = err_tb

                    row["candidate_success"] = False
                    row["candidate_error_type"] = err_type
                    row["candidate_error_message"] = err_msg
                    row["candidate_error_traceback"] = err_tb
                    row.update(_nan_candidate_metrics())

                    self.progress.log(
                        "\n[Warning] Candidate failed.\n"
                        f"  input: {self.input_path.name}\n"
                        f"  candidate_id: {i}\n"
                        f"  cluster_id: {cand.cluster_id}\n"
                        f"  error: {err_type}: {err_msg}"
                    )

                    rows.append(row)
                    # Save immediately after the failed tower as well.
                    self._write_candidate_summary(rows, summary_path)

                    if not self.cfg.continue_on_candidate_error:
                        raise

                    continue

                rows.append(row)
                # Important for long lines: save after every tower.
                self._write_candidate_summary(rows, summary_path)

        with self.progress.stage("Save summary"):
            line_success = not partial_line

            # Save scene-level visualization results after preprocessing.
            # 0 denotes background; 1,2,... denote candidate/tower instance ids.
            if self.cfg.save_point_clouds:
                self._save_txt(
                    self._output_path("scene_localization_result.txt"),
                    scene,
                    localization_scene_labels,
                )
                self._save_txt(
                    self._output_path("scene_precise_extraction_result.txt"),
                    scene,
                    precise_scene_labels,
                )

            # Save all candidate regions obtained after candidate tower localization.
            # This file is useful for visualizing the localization stage in the paper.
            if self.cfg.save_point_clouds and all_localized_points:
                self._save_txt(
                    self._output_path("all_localized_candidates.txt"),
                    np.vstack(all_localized_points),
                    np.concatenate(all_localized_labels),
                )

            # Save the successfully extracted tower points whenever available.
            # If some candidates fail, this file contains only successful candidates.
            if self.cfg.save_point_clouds and all_tower_points:
                self._save_txt(
                    self._output_path("all_towers.txt"),
                    np.vstack(all_tower_points),
                    np.concatenate(all_tower_labels),
                )

            # Re-save candidate-level summary to make sure the final version exists.
            self._write_candidate_summary(rows, summary_path)

            scene_metrics = None
            metric_scope = "not_available"
            evaluated_points = 0

            if scene_labels is not None:
                if line_success:
                    # Full-line metrics when all candidate towers are successfully processed.
                    y_true = scene_labels.astype(int)
                    y_pred = scene_pred
                    metric_scope = "full_line"
                    evaluated_points = len(scene_labels)
                else:
                    # Partial-line metrics: only points belonging to successfully processed
                    # candidate towers are evaluated. Failed candidates are excluded rather
                    # than being counted as false negatives.
                    valid_mask = valid_scene_pred_mask
                    evaluated_points = int(valid_mask.sum())
                    if evaluated_points > 0:
                        y_true = scene_labels[valid_mask].astype(int)
                        y_pred = scene_pred[valid_mask]
                        metric_scope = "successful_candidates_only"
                    else:
                        y_true = None
                        y_pred = None
                        metric_scope = "no_successful_candidate"

                if y_true is not None:
                    scene_metrics = binary_metrics(y_true, y_pred)
                    pd.DataFrame([{
                        "input_file": self.input_path.name,
                        "input_stem": self.input_stem,
                        "output_prefix": self.file_prefix,
                        "line_success": line_success,
                        "metric_scope": metric_scope,
                        "evaluated_points": evaluated_points,
                        "total_scene_points": len(scene_labels),
                        "successful_candidates": successful_candidates,
                        "failed_candidates": failed_candidates,
                        **scene_metrics,
                    }]).to_csv(scene_metrics_path, index=False)

                    if line_success:
                        self.progress.log(f"  - scene_metrics: {scene_metrics}")
                    else:
                        self.progress.log(
                            "  - partial scene_metrics computed using successfully processed candidates only: "
                            f"{scene_metrics}"
                        )
                else:
                    scene_metrics_path = None
                    self.progress.log(
                        "  - scene metrics skipped because no candidate tower was successfully processed."
                    )

            self.progress.log(f"  - candidate summary saved to: {summary_path}")
            if scene_metrics_path is not None:
                self.progress.log(f"  - scene metrics saved to: {scene_metrics_path}")

        return {
            "num_candidates": len(candidates),
            "successful_candidates": successful_candidates,
            "failed_candidates": failed_candidates,
            "line_success": line_success,
            "partial_line": partial_line,
            "first_error_type": first_error_type,
            "first_error_message": first_error_message,
            "first_error_traceback": first_error_traceback,
            "summary": rows,
            "scene_metrics": scene_metrics,
            "summary_path": str(summary_path),
            "candidate_summary_path": str(summary_path),
            "scene_metrics_path": "" if scene_metrics_path is None else str(scene_metrics_path),
            "all_localized_candidates_path": str(self._output_path("all_localized_candidates.txt")),
            "scene_localization_result_path": str(self._output_path("scene_localization_result.txt")),
            "scene_precise_extraction_result_path": str(self._output_path("scene_precise_extraction_result.txt")),
            "output_format": "txt",
            "output_prefix": self.file_prefix,
            "stages": {
                "preprocess": self.cfg.stages.preprocess,
                "localization": self.cfg.stages.localization,
                "precise_extraction": self.cfg.stages.precise_extraction,
            },
            "integer_internal": self.cfg.coordinates.integer_internal,
            "coordinate_scale": self.cfg.coordinates.scale,
            "continue_on_candidate_error": self.cfg.continue_on_candidate_error,
        }

    def _save_txt(self, path: Path, points: np.ndarray, labels: Optional[np.ndarray]) -> None:
        save_point_cloud(
            path,
            points,
            labels,
            integer_internal=self.cfg.coordinates.integer_internal,
            scale=self.cfg.coordinates.scale,
        )

    def _run_preprocess(self, points: np.ndarray, labels: Optional[np.ndarray]) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if not self.cfg.stages.preprocess:
            self.progress.log("\n[Stage] Preprocess skipped.")
            return points.copy(), None if labels is None else labels.copy()

        with self.progress.stage("Preprocess"):
            pre = self.icfg.preprocess
            scene, scene_labels = voxel_downsample(
                points,
                pre.voxel_size,
                labels,
                show_progress=self.cfg.progress.enabled,
            )
            scene, scene_labels, _ = statistical_outlier_removal(
                scene,
                pre.sor_neighbors,
                pre.sor_std_ratio,
                scene_labels,
                show_progress=self.cfg.progress.enabled,
                open3d_progress=self.cfg.progress.open3d_progress,
            )
        return scene, scene_labels

    def _run_localization(
        self,
        scene: np.ndarray,
        scene_labels: Optional[np.ndarray],
        scene_indices: np.ndarray,
    ):
        if not self.cfg.stages.localization:
            self.progress.log("\n[Stage] Localization skipped. The whole scene is treated as one candidate.")
            return [
                CandidateTower(
                    points=scene,
                    labels=scene_labels,
                    source_indices=scene_indices,
                    bbox_min=scene.min(axis=0),
                    bbox_max=scene.max(axis=0),
                    cluster_id=0,
                )
            ]

        with self.progress.stage("Localization"):
            pre = self.icfg.preprocess
            loc = self.icfg.localization
            dbscan_points, dbscan_labels = voxel_downsample(
                scene,
                loc.dbscan_voxel_size,
                scene_labels,
                show_progress=self.cfg.progress.enabled,
            )
            dbscan_points, dbscan_labels, _ = grid_near_ground_suppression(
                dbscan_points,
                pre.grid_size,
                pre.near_ground_height,
                dbscan_labels,
                show_progress=self.cfg.progress.enabled,
            )

            candidates = localize_candidate_towers(
                original_points=scene,
                dbscan_points=dbscan_points,
                eps=loc.dbscan_eps,
                min_points=loc.dbscan_min_points,
                min_cluster_height=loc.min_cluster_height,
                bbox_expand_xy=loc.bbox_expand_xy,
                original_labels=scene_labels,
                original_indices=scene_indices,
                show_progress=self.cfg.progress.enabled,
                open3d_progress=self.cfg.progress.open3d_progress,
            )
        return candidates
