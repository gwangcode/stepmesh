"""Shared fixtures for stepmesh tests.

Examples directory resolution order:
    1. $STEPMESH_EXAMPLES_DIR
    2. <repo>/examples/
    3. <repo>/../georay/examples/
    4. common Colab paths
"""
import os
import sys
from pathlib import Path

import pytest


# 确保 stepmesh 可以被 import（不依赖 pip install -e .）
_HERE = Path(__file__).resolve()
_REPO = _HERE.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def _find_examples_dir():
    env = os.environ.get("STEPMESH_EXAMPLES_DIR")
    if env:
        p = Path(env)
        if p.is_dir() and any(p.glob("*.step")):
            return p

    candidates = [
        _REPO / "examples",
        _REPO.parent / "georay" / "examples",
        _REPO.parent / "examples",
        Path("/content/georay/examples"),
        Path("/content/georay_check/georay/examples"),
    ]
    for c in candidates:
        if c.is_dir() and any(c.glob("*.step")):
            return c
    return None


@pytest.fixture(scope="session")
def examples_dir():
    d = _find_examples_dir()
    if d is None:
        pytest.skip(
            "No STEP examples directory found. "
            "Set STEPMESH_EXAMPLES_DIR or place .step files in "
            "stepmesh/examples/"
        )
    return d


@pytest.fixture(scope="session")
def all_steps(examples_dir):
    files = sorted(str(p) for p in examples_dir.glob("*.step"))
    if not files:
        pytest.skip(f"No .step files in {examples_dir}")
    return files


@pytest.fixture(scope="session")
def standard_trio(examples_dir):
    """cone + cube + cylinder, at least two of them if available."""
    names = ["cone.step", "cube.step", "cylinder.step"]
    files = []
    for n in names:
        p = examples_dir / n
        if p.exists():
            files.append(str(p))
    if len(files) < 2:
        pytest.skip("Need at least 2 standard example STEP files")
    return files


@pytest.fixture(autouse=True)
def cleanup_gmsh():
    """Ensure gmsh is finalized after each test, even on failure."""
    yield
    try:
        import gmsh
        if gmsh.is_initialized():
            gmsh.finalize()
    except Exception:
        pass
