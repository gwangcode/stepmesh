"""Tests for stepmesh mesh generation and format conversion.

Complements tests/test_glue.py (which focuses on glue()'s watertightness
and mapping semantics). This file covers:

    - glue() basic shape / dtype / consistency
    - step2msh() writing a .msh
    - vf2stl()       vertices/faces -> STL
    - msh2stl()      .msh -> STL
    - msh2unv()      .msh -> UNV

Run from the repository root:

    pytest tests/test_mesh.py -v

Examples directory resolution is handled by conftest.py.
"""
import os
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from stepmesh import glue, step2msh, vf2stl, msh2stl, msh2unv


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_step_paths(examples_dir):
    """Return a list of at least two STEP files from the examples dir."""
    files = sorted(str(p) for p in examples_dir.glob("*.step"))
    if len(files) < 2:
        pytest.skip(f"Need >= 2 STEP files in {examples_dir}")
    return files


@pytest.fixture
def trio_paths(examples_dir):
    """Return cone/cube/cylinder if present, else skip."""
    names = ["cone.step", "cube.step", "cylinder.step"]
    files = [str(examples_dir / n) for n in names
             if (examples_dir / n).exists()]
    if len(files) < 2:
        pytest.skip("Need at least 2 of cone/cube/cylinder")
    return files


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def edge_counts(faces):
    c = Counter()
    for tri in faces:
        for i in range(3):
            a, b = int(tri[i]), int(tri[(i + 1) % 3])
            if a == b:
                continue
            e = (a, b) if a < b else (b, a)
            c[e] += 1
    return c


def assert_nonmanifold_free(v, f):
    ec = edge_counts(f)
    assert all(c == 2 for c in ec.values()), \
        f"edges with count != 2: {sum(1 for c in ec.values() if c != 2)}"


def load_stl_watertight(stl_path):
    """Load an STL file and restore topology by merging coincident vertices.

    STL stores only raw triangle vertex coordinates with no vertex-sharing
    information; every loader must merge coincident vertices to recover
    the underlying topology. ``process=False`` disables that, so we do it
    explicitly via ``merge_vertices()``.
    """
    trimesh = pytest.importorskip("trimesh")
    m = trimesh.load(str(stl_path), process=False)
    m.merge_vertices()
    return m


# ---------------------------------------------------------------------------
# glue() basic
# ---------------------------------------------------------------------------

def test_glue_basic(sample_step_paths):
    """glue() returns 5-tuple with correct shapes and dtypes."""
    result = glue(
        sample_step_paths,
        min_size=1.0,
        max_size=3.0,
        return_internal_surfaces=False,
    )
    assert len(result) == 5
    vertices, faces, mapping_table, file_to_ids_dict, tet_data = result

    # ---- types ----
    assert isinstance(vertices, np.ndarray)
    assert isinstance(faces, np.ndarray)
    assert isinstance(mapping_table, np.ndarray)
    assert isinstance(file_to_ids_dict, dict)
    assert isinstance(tet_data, list)

    # ---- dtypes ----
    assert vertices.dtype == np.float64
    assert faces.dtype == np.int64
    assert mapping_table.dtype == np.int32

    # ---- shapes ----
    assert vertices.ndim == 2 and vertices.shape[1] == 3
    assert faces.ndim == 2 and faces.shape[1] == 3
    assert mapping_table.shape == (len(faces), 3)
    assert len(vertices) > 0
    assert len(faces) > 0

    # ---- face index range ----
    assert faces.min() >= 0
    assert faces.max() < len(vertices)

    # ---- file_to_ids covers all inputs ----
    expected = {os.path.basename(p) for p in sample_step_paths}
    assert set(file_to_ids_dict.keys()) == expected
    for fname, ids in file_to_ids_dict.items():
        assert isinstance(ids, list)
        assert len(ids) >= 1


def test_glue_returns_empty_tet_data_by_default(trio_paths):
    *_, tet_data = glue(
        trio_paths, min_size=0.5, max_size=2.0,
        return_internal_surfaces=False,
    )
    assert tet_data == []


def test_glue_tet_data_when_requested(trio_paths):
    *_, tet_data = glue(
        trio_paths, min_size=0.5, max_size=2.0,
        return_each_tet_as_body=True,
    )
    assert len(tet_data) > 0
    required = {"tet_id", "centroid", "volume_tag", "filename"}
    for d in tet_data:
        assert required.issubset(d.keys())
        assert np.asarray(d["centroid"]).shape == (3,)


def test_glue_internal_surfaces_have_neg_vol(trio_paths):
    """With return_internal_surfaces=True, mapping_table[:, 1] > 0 marks
    interface faces."""
    _, _, mt, _, _ = glue(
        trio_paths, min_size=0.3, max_size=1.0,
        return_internal_surfaces=True,
    )
    assert (mt[:, 1] > 0).any()
    # pos != neg on interface faces
    iface = mt[:, 1] > 0
    assert not (mt[iface, 0] == mt[iface, 1]).any()


def test_glue_outer_only_watertight(trio_paths):
    v, f, *_ = glue(
        trio_paths, min_size=0.3, max_size=1.0,
        return_internal_surfaces=False,
    )
    assert_nonmanifold_free(v, f)


def test_glue_single_file(examples_dir):
    p = examples_dir / "cube.step"
    if not p.exists():
        pytest.skip("cube.step not found")
    v, f, mt, fid, _ = glue(
        [str(p)], min_size=0.3, max_size=1.5,
        return_internal_surfaces=False,
    )
    assert len(v) > 0
    assert len(f) > 0
    assert list(fid.keys()) == ["cube.step"]
    assert_nonmanifold_free(v, f)


# ---------------------------------------------------------------------------
# step2msh
# ---------------------------------------------------------------------------

def test_step2msh_writes_msh(tmp_path, sample_step_paths):
    out = tmp_path / "multi.msh"
    fid = step2msh(
        sample_step_paths,
        str(out),
        min_size=1.0,
        max_size=3.0,
    )
    assert out.exists()
    assert out.stat().st_size > 0

    expected = {os.path.basename(p) for p in sample_step_paths}
    assert set(fid.keys()) == expected
    for fname, ids in fid.items():
        assert len(ids) == 1, f"{fname}: {ids}"


def test_step2msh_physical_groups_persist(tmp_path, trio_paths):
    """Reload the .msh and verify physical volume groups are present."""
    import gmsh
    out = tmp_path / "multi.msh"
    step2msh(trio_paths, str(out), min_size=1.0, max_size=3.0)

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.open(str(out))
    phys = gmsh.model.getPhysicalGroups(dim=3)
    gmsh.finalize()
    assert len(phys) >= 1


# ---------------------------------------------------------------------------
# Format conversions
# ---------------------------------------------------------------------------

def test_format_conversions(tmp_path, sample_step_paths):
    """glue -> vf2stl -> (step2msh) -> msh2stl / msh2unv end-to-end."""
    # glue returns 5-tuple
    vertices, faces, _, _, _ = glue(
        sample_step_paths,
        min_size=1.0,
        max_size=3.0,
        return_internal_surfaces=False,
    )

    # ---- vf2stl (binary + ascii) ----
    stl_bin = tmp_path / "out_bin.stl"
    vf2stl(vertices, faces, str(stl_bin), binary=True)
    assert stl_bin.exists() and stl_bin.stat().st_size > 0

    stl_ascii = tmp_path / "out_ascii.stl"
    vf2stl(vertices, faces, str(stl_ascii), binary=False)
    assert stl_ascii.exists() and stl_ascii.stat().st_size > 0
    text = stl_ascii.read_text()
    assert "solid" in text and "endsolid" in text

    # ---- step2msh ----
    msh_path = tmp_path / "test_out.msh"
    step2msh(
        sample_step_paths, str(msh_path),
        min_size=1.0, max_size=3.0,
    )
    assert msh_path.exists() and msh_path.stat().st_size > 0

    # ---- msh2stl ----
    stl_from_msh = tmp_path / "from_msh.stl"
    msh2stl(str(msh_path), str(stl_from_msh))
    assert stl_from_msh.exists() and stl_from_msh.stat().st_size > 0
    text = stl_from_msh.read_text()
    assert "solid" in text and "endsolid" in text

    # ---- msh2unv ----
    unv_path = tmp_path / "out.unv"
    msh2unv(str(msh_path), str(unv_path))
    assert unv_path.exists() and unv_path.stat().st_size > 0


def test_vf2stl_binary_roundtrip(tmp_path, trio_paths):
    """Binary STL should be parseable by trimesh."""
    trimesh = pytest.importorskip("trimesh")
    v, f, *_ = glue(
        trio_paths, min_size=0.3, max_size=1.0,
        return_internal_surfaces=False,
    )
    out = tmp_path / "mesh.stl"
    vf2stl(v, f, str(out), binary=True)
    m = trimesh.load(str(out), process=False)
    assert len(m.faces) == len(f)


def test_msh2stl_produces_watertight(tmp_path, trio_paths):
    """msh2stl output, after STL vertex merging, must be watertight.

    STL does not store topology, so the loader must merge coincident
    vertices to reconstruct connectivity. We do that explicitly.
    """
    msh = tmp_path / "m.msh"
    step2msh(trio_paths, str(msh), min_size=0.3, max_size=1.0)
    stl = tmp_path / "m.stl"
    msh2stl(str(msh), str(stl))

    m = load_stl_watertight(stl)
    assert m.is_watertight, (
        f"after merge_vertices: V={len(m.vertices)} F={len(m.faces)} "
        f"euler={m.euler_number}"
    )


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

def test_step2msh_missing_file(tmp_path):
    out = tmp_path / "x.msh"
    # No STEP files -> should raise
    with pytest.raises(ValueError):
        step2msh([], str(out))


def test_msh2stl_missing_input(tmp_path):
    with pytest.raises(FileNotFoundError):
        msh2stl(str(tmp_path / "nope.msh"), str(tmp_path / "x.stl"))


def test_msh2unv_missing_input(tmp_path):
    with pytest.raises(FileNotFoundError):
        msh2unv(str(tmp_path / "nope.msh"), str(tmp_path / "x.unv"))


def test_msh2stl_include_internal_adds_faces(tmp_path, trio_paths):
    """include_internal=True should output boundary + internal faces,
    and the count must be strictly larger than boundary-only."""
    msh = tmp_path / "m.msh"
    step2msh(trio_paths, str(msh), min_size=0.3, max_size=1.0)

    stl_outer = tmp_path / "outer.stl"
    stl_full = tmp_path / "full.stl"
    msh2stl(str(msh), str(stl_outer), include_internal=False)
    msh2stl(str(msh), str(stl_full), include_internal=True)

    # 计数 STL 里的 "facet normal"
    def count_facets(path):
        return path.read_text().count("facet normal")

    n_outer = count_facets(stl_outer)
    n_full = count_facets(stl_full)

    assert n_outer > 0
    assert n_full > n_outer, (
        f"include_internal=True should add faces: "
        f"outer={n_outer}, full={n_full}"
    )


def test_msh2stl_internal_contains_contact_geometry(tmp_path, trio_paths):
    """The extra faces in include_internal=True must include the cube top
    (z=10) and cube bottom (z=0) contact disks, not just duplicated outer
    faces."""
    import trimesh
    msh = tmp_path / "m.msh"
    step2msh(trio_paths, str(msh), min_size=0.3, max_size=1.0)

    stl_outer = tmp_path / "outer.stl"
    stl_full = tmp_path / "full.stl"
    msh2stl(str(msh), str(stl_outer), include_internal=False)
    msh2stl(str(msh), str(stl_full), include_internal=True)

    m_outer = trimesh.load(str(stl_outer), process=False)
    m_full = trimesh.load(str(stl_full), process=False)

    # 内部面应包含平面 (z=0 圆盘 + z=10 圆盘)
    # 检查 full 里有不在 outer 的三角形，且其顶点 z 都接近 0 或 10
    def tri_keys(mesh):
        keys = set()
        for tri in mesh.faces:
            pts = tuple(sorted(tuple(np.round(mesh.vertices[i], 6))
                               for i in tri))
            keys.add(pts)
        return keys

    outer_keys = tri_keys(m_outer)
    full_keys = tri_keys(m_full)
    extra = full_keys - outer_keys

    assert len(extra) > 0, "no extra faces in include_internal=True"

    # 额外面里应该有位于 z=0 或 z=10 平面上的
    n_at_z_plane = 0
    for tri_pts in extra:
        zs = [p[2] for p in tri_pts]
        if all(abs(z - 0.0) < 1e-4 for z in zs):
            n_at_z_plane += 1
        elif all(abs(z - 10.0) < 1e-4 for z in zs):
            n_at_z_plane += 1
    assert n_at_z_plane > 0, (
        f"expected extra faces on z=0 or z=10 contact planes, "
        f"got {n_at_z_plane} / {len(extra)}"
    )
