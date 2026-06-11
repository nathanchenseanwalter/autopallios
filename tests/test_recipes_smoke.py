"""The literal "everything runs on Day 1" guarantee: both recipes run end-to-end on mock data."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_recipe(rel_path: str, out_dir: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(REPO_ROOT / rel_path), "--out", str(out_dir), "--frames", "3"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=600,
    )


def test_wound_healing_recipe_runs(tmp_path):
    proc = _run_recipe("recipes/wound_healing/run_migration.py", tmp_path)
    assert proc.returncode == 0, proc.stderr
    assert "USING MOCK DATA" in proc.stdout
    # Debug mode writes a mask sequence and a temporal-consistency table.
    assert list(tmp_path.glob("migration_*/masks/*.tif"))
    assert list(tmp_path.glob("migration_*/tables/temporal_consistency_summary.csv"))


def test_kill_curve_recipe_runs(tmp_path):
    proc = _run_recipe("recipes/cancer_cell_death/run_kill_curve.py", tmp_path)
    assert proc.returncode == 0, proc.stderr
    assert "USING MOCK DATA" in proc.stdout
    # Production mode (debug=False) writes NO mask TIFFs ...
    assert not list(tmp_path.glob("kill_curve_*/masks/*.tif"))
    # ... but does write the supervised benchmark tables.
    assert list(tmp_path.glob("kill_curve_*/tables/supervised_aggregate.csv"))
