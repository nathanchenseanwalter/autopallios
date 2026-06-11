"""Guard the no-path-hacks rule: every public import resolves after an editable install."""

from __future__ import annotations


def test_top_level_import_and_version():
    import autopallios

    assert isinstance(autopallios.__version__, str)
    assert autopallios.Pipeline is not None  # lazily exposed


def test_core_submodule_imports():
    from autopallios.core import baseline, filter, io, segmenter  # noqa: F401

    assert hasattr(segmenter, "Segmenter")
    assert hasattr(io, "load")
    assert hasattr(baseline, "BaselineSegmenter")
    assert hasattr(filter, "ArtifactFilter")


def test_modules_imports():
    from autopallios.modules import evaluation, intensity, tracking

    assert hasattr(tracking, "Tracker")
    assert hasattr(intensity, "IntensityAnalyzer")
    assert hasattr(evaluation, "SupervisedMetrics")
    assert hasattr(evaluation, "UnsupervisedMetrics")


def test_data_and_recipe_imports():
    from autopallios.data import synthetic
    from autopallios.recipe import RecipeContext, resolve_or_mock  # noqa: F401

    assert hasattr(synthetic, "make_movie_with_labels")
