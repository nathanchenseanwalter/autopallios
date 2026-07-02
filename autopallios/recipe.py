"""Glue every recipe imports, so recipes stay short, uniform, and reproducible.

Two helpers:

- :class:`RecipeContext`, creates a tidy, timestamped ``output/`` folder next to your
  recipe (``masks/``, ``tables/``, ``figures/``), and writes a ``manifest.json`` recording
  exactly how the run was configured (data source, debug flag, seed, git commit). That
  manifest is what makes the Week-4 validation study credible, every number is traceable.

- :func:`resolve_or_mock`, load real data if it's there, otherwise fabricate a synthetic
  movie so the recipe *always runs* (Day 1, no real files, no GPU). It announces loudly
  which path it took, so no one mistakes synthetic results for real ones.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

#: Sentinel meaning "the caller did not ask for ground truth" (distinct from ``None``).
_UNSET = object()


def _git_sha() -> str | None:
    """Best-effort short git commit hash for provenance (``None`` if unavailable)."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except Exception:
        return None


class RecipeContext:
    """A per-run output folder + reproducibility manifest, created next to a recipe.

    Args:
        recipe_file: Pass ``__file__`` from the recipe, output lands beside it.
        experiment: Short name for this run (folded into the run folder name).
        debug: Whether this run is in debug mode (recorded; also what gates mask dumps).
        seed: RNG seed recorded for reproducibility.

    Attributes:
        run_dir: ``output/<experiment>_<timestamp>/`` for this run.
        masks_dir / tables_dir / figures_dir: subfolders for each kind of artifact.
        manifest_path: path to ``manifest.json``.
    """

    def __init__(
        self,
        recipe_file: str,
        experiment: str,
        debug: bool,
        seed: int = 0,
        output_root: str | Path | None = None,
    ) -> None:
        self.experiment = experiment
        self.debug = bool(debug)
        self.seed = int(seed)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        base = Path(output_root) if output_root else Path(recipe_file).resolve().parent / "output"
        self.run_dir = base / f"{experiment}_{stamp}"
        self.masks_dir = self.run_dir / "masks"
        self.tables_dir = self.run_dir / "tables"
        self.figures_dir = self.run_dir / "figures"
        for d in (self.masks_dir, self.tables_dir, self.figures_dir):
            d.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.run_dir / "manifest.json"
        self._manifest: dict[str, Any] = {
            "experiment": experiment,
            "debug": self.debug,
            "seed": self.seed,
            "git_sha": _git_sha(),
            "created": stamp,
            "data_source": None,
            "model": None,
        }
        self._write_manifest()

    def _write_manifest(self) -> None:
        self.manifest_path.write_text(json.dumps(self._manifest, indent=2))

    def update_manifest(self, **fields: Any) -> None:
        """Record extra fields (e.g. ``model="cellpose"``) into ``manifest.json``."""
        self._manifest.update(fields)
        self._write_manifest()

    def save_table(self, df: pd.DataFrame, name: str) -> Path:
        """Write a DataFrame to ``tables/<name>`` (CSV) and return its path."""
        path = self.tables_dir / name
        df.to_csv(path, index=False)
        return path

    def save_figure(self, fig, name: str) -> Path:
        """Write a matplotlib figure to ``figures/<name>`` and return its path."""
        path = self.figures_dir / name
        fig.savefig(path, dpi=120, bbox_inches="tight")
        return path


def _load_label_dir(path: str | Path) -> np.ndarray:
    """Load a directory (or file) of label TIFFs as a ``(T, H, W)`` int32 stack."""
    from .core.io import load

    arr = load(path, kind="auto", as_gray=True)
    return arr[..., 0].astype(np.int32)


def resolve_or_mock(
    real_path: str | Path | None,
    kind: str,
    ctx: RecipeContext | None = None,
    *,
    loader_kwargs: dict | None = None,
    mock_kwargs: dict | None = None,
    with_ground_truth: Any = _UNSET,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Return real data if available, else a synthetic scene, so a recipe always runs.

    Args:
        real_path: Path to real data, or ``None`` to force synthetic.
        kind: A scene name from :data:`autopallios.data.synthetic.SCENES` (e.g.
            ``"mock_migration"``), used when fabricating.
        ctx: The :class:`RecipeContext` (its manifest records which source was used).
        loader_kwargs: Extra args for :func:`autopallios.core.io.load` (e.g.
            ``kind="directory"``, ``pattern=...``, ``as_gray=True``).
        mock_kwargs: Extra args for the synthetic generator (e.g. ``n_frames=20``).
        with_ground_truth: Omit it to get back just the movie. Pass it (a path or
            ``None``) to also get ground-truth labels, returns ``(movie, labels)``.

    Returns:
        ``movie`` (``(T, H, W, C)``), or ``(movie, labels)`` if ``with_ground_truth`` was passed.
    """
    from .core.io import load
    from .data import synthetic

    loader_kwargs = loader_kwargs or {}
    mock_kwargs = mock_kwargs or {}
    wants_gt = with_ground_truth is not _UNSET

    use_real = real_path is not None and Path(real_path).exists()
    if use_real:
        source = str(real_path)
        movie = load(real_path, **loader_kwargs)
        print(f"[autopallios] Using REAL data: {source}")
        if not wants_gt:
            _record(ctx, source)
            return movie
        gt_path = with_ground_truth
        if gt_path is None or not Path(gt_path).exists():
            raise FileNotFoundError(
                "Real data was found but no ground-truth labels were provided. "
                "Set the ground-truth path (GT_DIR), or run without real data to use "
                "synthetic ground truth."
            )
        _record(ctx, source, gt=str(gt_path))
        return movie, _load_label_dir(gt_path)

    # --- synthetic fallback ---
    source = f"mock:{kind}"
    print(f"[autopallios] USING MOCK DATA (scene={kind!r}), no real files found.")
    if not wants_gt:
        _record(ctx, source)
        return synthetic.make_scene(kind, **mock_kwargs)
    movie, labels = synthetic.make_scene_with_labels(kind, **mock_kwargs)
    _record(ctx, source, gt="mock")
    return movie, labels


def _record(ctx: RecipeContext | None, source: str, gt: str | None = None) -> None:
    if ctx is not None:
        ctx.update_manifest(data_source=source, ground_truth=gt)


__all__ = ["RecipeContext", "resolve_or_mock"]
