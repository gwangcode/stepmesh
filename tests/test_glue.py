"""Tests for stepmesh.glue, step2msh, vf2stl.

Run from the repository root with:

    pytest tests/ -v

Optional environment variable:

    STEPMESH_EXAMPLES_DIR=/path/to/examples
"""
import os
from collections import Counter

import numpy as np
import pytest

from stepmesh import glue, step2msh, vf2stl


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def edge_stats(faces):
    """Return (boundary_count, nonmanifold_count)."""
    ec = Counter()
    for tri in faces:
        n = len(tri)
        for i in range(n):
            a, b = int(tri[i]), int(tri[(i + 1) % n])
            if a == b:
                continue
            e = (a, b) if a < b else (b, a)
            ec[e] += 1
    bd = sum(1 for c in ec.values() if c == 1)
    nm = sum(1 for c in ec.values() if c > 2)
    return bd, nm


def assert_watertight(v, f):
    """Assert that the triangle mesh is a closed 2-manifold."""
    assert len(v) > 0
    assert len(f) > 0
    assert f.shape[1] == 3
    assert f.min() >= 0
    assert f.max() < len(v)
    for tri in f:
        assert (tri[0] != tri[1] and tri[1] != tri[2] and tri[0] != tri[2]), \
            "degenerate triangle"
    bd, nm = edge_stats(f)
    assert bd == 0, f"boundary edges = {bd}"
    assert nm == 0, f"nonmanifold edges = {nm}"


def assert_mapping_table(mt, f, fid, expect_interface):
    assert mt.shape == (len(f), 3)
    np.testing.assert_array_equal(
        mt[:, 2], np.arange(len(f), dtype=mt.dtype)
    )
    valid = set()
    for vs in fid.values():
        valid.update(vs)
    assert set(mt[:, 0].tolist()).issubset(valid), \
        f"pos_vol out of range: {set(mt[:, 0].tolist()) - valid}"
    assert (set(mt[:, 1].tolist()) - {0}).issubset(valid), \
        f"neg_vol out of range: {set(mt[:, 1].tolist()) - {0} - valid}"

    iface = mt[:, 1] > 0
    if expect_interface:
        assert iface.any(), "expected shared interface faces"
        assert not (mt[iface, 0] == mt[iface, 1]).any(), \
            "interface face has pos_vol == neg_vol"
    else:
        assert not iface.any(), \
            f"unexpected interface faces: {int(iface.sum())}"


# ---------------------------------------------------------------------------
# Outer surface only: must be watertight
# ---------------------------------------------------------------------------

def test_outer_only_watertight(all_steps):
    v, f, mt, fid, tc = glue(
        all_steps, min_size=0.1, max_size=0.5,
        return_internal_surfaces=False,
    )
    assert_watertight(v, f)


def test_outer_only_trimesh_watertight(standard_trio):
    trimesh = pytest.importorskip("trimesh")
    v, f, *_ = glue(
        standard_trio, min_size=0.1, max_size=0.5,
        return_internal_surfaces=False,
    )
    m = trimesh.Trimesh(vertices=v, faces=f, process=False)
    assert m.is_watertight


def test_outer_only_no_interface_faces(standard_trio):
    v, f, mt, fid, _ = glue(
        standard_trio, min_size=0.2, max_size=1.0,
        return_internal_surfaces=False,
    )
    assert_mapping_table(mt, f, fid, expect_interface=False)


def test_outer_only_tet_data_empty(standard_trio):
    *_, tc = glue(
        standard_trio, min_size=0.3, max_size=1.0,
        return_internal_surfaces=False,
    )
    assert tc == []


# ---------------------------------------------------------------------------
# With internal interfaces
# ---------------------------------------------------------------------------

def test_internal_has_interface(standard_trio):
    v, f, mt, fid, _ = glue(
        standard_trio, min_size=0.2, max_size=1.0,
        return_internal_surfaces=True,
    )
    assert_mapping_table(mt, f, fid, expect_interface=True)


def test_internal_no_boundary(standard_trio):
    v, f, *_ = glue(
        standard_trio, min_size=0.2, max_size=1.0,
        return_internal_surfaces=True,
    )
    bd, _ = edge_stats(f)
    assert bd == 0


def test_true_minus_false_equals_interface_count(standard_trio):
    _, f_false, *_ = glue(
        standard_trio, min_size=0.2, max_size=1.0,
        return_internal_surfaces=False,
    )
    _, f_true, mt_true, _, _ = glue(
        standard_trio, min_size=0.2, max_size=1.0,
        return_internal_surfaces=True,
    )
    n_iface = int((mt_true[:, 1] > 0).sum())
    assert len(f_true) - len(f_false) == n_iface


# ---------------------------------------------------------------------------
# file_to_ids_dict
# ---------------------------------------------------------------------------

def test_file_to_ids_keys_match_inputs(all_steps):
    *_, fid, _ = glue(
        all_steps, min_size=0.3, max_size=1.0,
        return_internal_surfaces=False,
    )
    want = {os.path.basename(p) for p in all_steps}
    assert set(fid.keys()) == want


def test_file_to_ids_unique(standard_trio):
    *_, fid, _ = glue(
        standard_trio, min_size=0.3, max_size=1.0,
        return_internal_surfaces=False,
    )
    all_ids = [i for ids in fid.values() for i in ids]
    assert len(all_ids) == len(set(all_ids))
    for fname, ids in fid.items():
        assert len(ids) == 1, f"{fname}: {ids}"


def test_file_to_ids_stable_between_modes(standard_trio):
    _, _, _, fid_false, _ = glue(
        standard_trio, min_size=0.3, max_size=1.0,
        return_internal_surfaces=False,
    )
    _, _, _, fid_true, _ = glue(
        standard_trio, min_size=0.3, max_size=1.0,
        return_internal_surfaces=True,
    )
    assert fid_false == fid_true


# ---------------------------------------------------------------------------
# Mesh sizes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("min_size,max_size", [
    (0.5, 2.0),
    (0.2, 1.0),
    (0.1, 0.5),
])
def test_mesh_size_watertight(all_steps, min_size, max_size):
    v, f, *_ = glue(
        all_steps, min_size=min_size, max_size=max_size,
        return_internal_surfaces=False,
    )
    assert_watertight(v, f)


def test_fine_mesh_watertight(all_steps):
    v, f, *_ = glue(
        all_steps, min_size=0.05, max_size=0.3,
        return_internal_surfaces=False,
    )
    assert_watertight(v, f)


# ---------------------------------------------------------------------------
# Subsets
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("subset", [
    ("cone.step", "cube.step"),
    ("cube.step", "cylinder.step"),
    ("cone.step",),
    ("cube.step",),
])
def test_subsets_watertight(examples_dir, subset):
    steps = []
    for name in subset:
        p = examples_dir / name
        if not p.exists():
            pytest.skip(f"{name} not found")
        steps.append(str(p))

    v, f, *_ = glue(steps, min_size=0.3, max_size=1.5,
                    return_internal_surfaces=False)
    assert_watertight(v, f)
    want = {os.path.basename(p) for p in steps}
    *_, fid, _ = glue(steps, min_size=0.3, max_size=1.5,
                      return_internal_surfaces=False)
    assert set(fid.keys()) == want


def test_single_file_true_equals_false(examples_dir):
    p = examples_dir / "cube.step"
    if not p.exists():
        pytest.skip("cube.step not found")
    _, f_false, *_ = glue([str(p)], min_size=0.3, max_size=1.5,
                          return_internal_surfaces=False)
    _, f_true, *_ = glue([str(p)], min_size=0.3, max_size=1.5,
                         return_internal_surfaces=True)
    assert len(f_true) == len(f_false)


# ---------------------------------------------------------------------------
# return_each_tet_as_body
# ---------------------------------------------------------------------------

def test_each_tet_as_body(all_steps):
    v, f, mt, fid, tc = glue(
        all_steps, min_size=0.5, max_size=2.0,
        return_each_tet_as_body=True,
    )
    assert len(tc) > 0
    required = {"tet_id", "centroid", "volume_tag", "filename"}
    tet_ids = set()
    for d in tc:
        assert required.issubset(d.keys()), f"missing keys in {d}"
        assert np.asarray(d["centroid"]).shape == (3,)
        assert d["filename"] in fid
        tet_ids.add(d["tet_id"])

    for fname, ids in fid.items():
        assert set(ids).issubset(tet_ids), \
            f"{fname}: ids not a subset of tet_ids"


# ---------------------------------------------------------------------------
# STL export
# ---------------------------------------------------------------------------

def test_vf2stl_binary(tmp_path, standard_trio):
    v, f, *_ = glue(standard_trio, min_size=0.3, max_size=1.0,
                    return_internal_surfaces=False)
    out = tmp_path / "mesh.stl"
    vf2stl(v, f, str(out), binary=True)
    assert out.exists()
    assert out.stat().st_size > 0


def test_vf2stl_ascii(tmp_path, standard_trio):
    v, f, *_ = glue(standard_trio, min_size=0.3, max_size=1.0,
                    return_internal_surfaces=False)
    out = tmp_path / "mesh_ascii.stl"
    vf2stl(v, f, str(out), binary=False)
    assert out.exists()
    text = out.read_text()
    assert "solid" in text
    assert "endsolid" in text


# ---------------------------------------------------------------------------
# step2msh
# ---------------------------------------------------------------------------

def test_step2msh_writes_file(tmp_path, all_steps):
    out = tmp_path / "multi.msh"
    fid = step2msh(all_steps, str(out), min_size=0.5, max_size=2.0)
    assert out.exists()
    assert out.stat().st_size > 0
    want = {os.path.basename(p) for p in all_steps}
    assert set(fid.keys()) == want


# ---------------------------------------------------------------------------
# Winding (known limitation, fixable with trimesh.repair.fix_normals)
# ---------------------------------------------------------------------------

def test_winding_can_be_made_consistent(standard_trio):
    """Winding across different gmsh surfaces is not globally consistent
    in the raw output; trimesh.repair.fix_normals() can fix it, and
    watertightness is preserved before and after."""
    trimesh = pytest.importorskip("trimesh")
    v, f, *_ = glue(standard_trio, min_size=0.3, max_size=1.0,
                    return_internal_surfaces=False)

    m = trimesh.Trimesh(vertices=v, faces=f, process=False)
    assert m.is_watertight

    # fix_normals should not break watertightness
    m.fix_normals()
    assert m.is_winding_consistent
    assert m.is_watertight
