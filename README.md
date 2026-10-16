# stepmesh 

[![DOI](https://zenodo.org/badge/1332405493.svg)](https://doi.org/10.5281/zenodo.21909276)

A Python package for processing 3D STEP/CAD files, performing multi-body boolean operations, and generating tetrahedral meshes (`.msh`), STL files (`.stl`), and UNV files (`.unv`) using **Gmsh** and **NumPy**.

**stepmesh** is designed for finite element preprocessing of multi-body CAD assemblies. It handles STEP import, multi-body boolean fragmentation, conformal mesh generation, and data extraction entirely in memory — no manual Gmsh or CAD software interaction required.

---

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [API Reference](#api-reference)
- [How Multi-Body Meshing Works](#how-multi-body-meshing-works)
- [Output Data Format](#output-data-format)
- [Testing](#testing)
- [Requirements](#requirements)
- [License](#license)
- [Citation](#citation)

---

## Features

- **Multi-body STEP meshing** — Automatic `fragment` + `removeAllDuplicates` operations across all imported bodies, producing matching conformal meshes on contact surfaces
- **In-memory data extraction** — Returns NumPy arrays directly (vertices, triangle faces, mapping table, tet centroids) without intermediate files
- **Format conversion** — Convert Gmsh `.msh` to `.unv` (compatible with FreeCAD FEM workbench), and export to binary or ASCII STL
- **Watertight outer surface** — With `return_internal_surfaces=False`, produces a fully watertight closed surface suitable for external flow, 3D printing, and visualization
- **Shared interface surfaces** — With `return_internal_surfaces=True`, preserves internal shared interfaces (one geometric surface referenced by both volumes) — ideal for conjugate heat transfer, fluid-structure interaction, and other multi-domain simulations

---

## Installation

Install from GitHub:

```bash
pip install git+https://github.com/gwangcode/stepmesh.git
```

Or clone and install in development mode:

```bash
git clone https://github.com/gwangcode/stepmesh.git
cd stepmesh
pip install -e .
```

### System Dependencies

Gmsh requires OpenGL/GLU libraries. On Ubuntu/Debian:

```bash
sudo apt-get install -y libglu1-mesa libgl1
```

On Google Colab:

```python
!apt-get install -y -qq libglu1-mesa libgl1
```

---

## Quick Start

### 1. Extract mesh data in memory

```python
from stepmesh import glue

vertices, faces, mapping_table, file_to_ids_dict, _ = glue(
    step_paths=["cube.step", "cylinder.step"],
    min_size=0.5,
    max_size=2.0,
    return_internal_surfaces=True,
)

print(f"Extracted {len(vertices)} vertices and {len(faces)} faces.")
print(f"File → Physical Volume ID mapping: {file_to_ids_dict}")
```

### 2. Write a `.msh` file directly

```python
from stepmesh import step2msh

file_to_ids = step2msh(
    step_paths=["cube.step", "cylinder.step"],
    output_msh_path="output.msh",
    min_size=0.5,
    max_size=2.0,
)
```

### 3. Format conversion

```python
from stepmesh import msh2stl, msh2unv

# Outer surface only → STL (watertight)
msh2stl("output.msh", "output_outer.stl", include_internal=False)

# Outer surface + internal contact interfaces → STL (multi-domain visualization)
msh2stl("output.msh", "output_full.stl", include_internal=True)

# → UNV (FreeCAD FEM)
msh2unv("output.msh", "output.unv")
```

### 4. Export STL from vertex/face arrays

```python
from stepmesh import glue, vf2stl

vertices, faces, _, _, _ = glue(
    ["cube.step", "cylinder.step"],
    min_size=0.5, max_size=2.0,
    return_internal_surfaces=False,
)

# Binary STL
vf2stl(vertices, faces, "mesh_binary.stl", binary=True)

# ASCII STL
vf2stl(vertices, faces, "mesh_ascii.stl", binary=False)
```

---

## API Reference

### `step2msh`

```python
step2msh(step_paths, output_msh_path, min_size=0.0, max_size=0.0)
```

Convert multiple STEP/CAD files into a 3D tetrahedral finite element mesh and write it to a `.msh` file. Performs boolean `fragment` across all bodies to ensure matching coplanar and contact surfaces, then merges geometrically coincident volumes/surfaces into shared interfaces. Each original CAD body becomes one Physical Volume.

| Parameter | Type | Description |
|---|---|---|
| `step_paths` | `list[str]` | Input CAD file paths. STEP, BREP, IGES, and other OCC-supported formats |
| `output_msh_path` | `str` | Output `.msh` file path |
| `min_size` | `float` | `Mesh.CharacteristicLengthMin`. 0 = Gmsh default |
| `max_size` | `float` | `Mesh.CharacteristicLengthMax`. 0 = Gmsh default |

**Returns**: `dict[str, list[int]]` — STEP filename → list of Physical Volume IDs (1-based).

---

### `glue`

```python
glue(step_paths, min_size=0.0, max_size=0.0,
     return_internal_surfaces=False, return_each_tet_as_body=False)
```

Generate a 3D tetrahedral mesh in memory, extract structured NumPy arrays, and establish ownership mapping back to source files.

| Parameter | Type | Description |
|---|---|---|
| `step_paths` | `list[str]` | List of STEP file paths |
| `min_size` | `float` | `Mesh.CharacteristicLengthMin` |
| `max_size` | `float` | `Mesh.CharacteristicLengthMax` |
| `return_internal_surfaces` | `bool` | `False` (default): export only the outer boundary skin (watertight). `True`: include internal shared interfaces |
| `return_each_tet_as_body` | `bool` | If `True`, emit per-tet centroid data |

**Returns**: 5-tuple `(vertices, faces, mapping_table, file_to_ids_dict, tet_centroids_data)`.

| Value | Type | Description |
|---|---|---|
| `vertices` | `np.ndarray` | `(N_verts, 3)` float64, unique 3D node coordinates |
| `faces` | `np.ndarray` | `(N_faces, 3)` int64, triangle vertex indices (0-based) |
| `mapping_table` | `np.ndarray` | `(N_faces, 3)` int32, see column meanings below |
| `file_to_ids_dict` | `dict[str, list[int]]` | STEP filename → Physical Volume IDs (or tet Element IDs) |
| `tet_centroids_data` | `list[dict]` | Per-tet centroid data. Non-empty only when `return_each_tet_as_body=True` |

**`mapping_table` columns**:

| Column | Name | Meaning |
|---|---|---|
| 0 | `pos_vol` | Positive attached Physical Volume ID (≥ 1) |
| 1 | `neg_vol` | Secondary attached Physical Volume ID: `0` = outer boundary face; `> 0` = shared interface face |
| 2 | `local_tri_id` | Row index of the face in the `faces` array |

**`return_internal_surfaces` semantics**:

| Value | Output | Watertight | Includes interfaces |
|---|---|---|---|
| `False` (default) | Outer surface only (faces shared by exactly 1 tet) | ✅ Fully watertight | ❌ |
| `True` | Outer surface + internal shared faces (shared by 2 tets, emitted once) | Interface edges are shared by 3 triangles → not watertight | ✅ |

---

### `vf2stl`

```python
vf2stl(vertices, faces, output_stl_path, binary=True)
```

Export vertex/face NumPy arrays directly to a standard STL file.

| Parameter | Type | Description |
|---|---|---|
| `vertices` | `np.ndarray` | `(N_verts, 3)` vertex coordinates |
| `faces` | `np.ndarray` | `(N_faces, 3)` triangle indices |
| `output_stl_path` | `str` | Output STL file path |
| `binary` | `bool` | `True` = binary STL, `False` = ASCII STL |

---

### `msh2unv`

```python
msh2unv(msh_path, unv_path)
```

Convert a Gmsh `.msh` file to an I-deas `.unv` file, compatible with the FreeCAD FEM workbench.

---

### `msh2stl`

```python
msh2stl(msh_path, stl_path, include_internal=False)
```

Extract triangle faces from a 3D tetrahedral `.msh` file and write an ASCII STL.

| Parameter | Type | Description |
|---|---|---|
| `msh_path` | `str` | Input `.msh` file path |
| `stl_path` | `str` | Output `.stl` file path |
| `include_internal` | `bool` | `False` (default): outer boundary only (watertight). `True`: include internal contact interfaces |

---

## How Multi-Body Meshing Works

`stepmesh` follows these steps inside `glue()` and `step2msh()`:

1. **Import** — Each STEP file is imported with `gmsh.model.occ.importShapes()`, which uses the OCC topology path. (`gmsh.merge()` uses the legacy path and makes subsequent `removeAllDuplicates()` ineffective.)
2. **Fragment** — Use `fragment([first], rest)` to split all mutually touching bodies. (`fragment(all, [])` does **not** split touching bodies in Gmsh.)
3. **Record centroids** — Save the centroid of every volume immediately after fragment.
4. **Deduplicate** — Call `removeAllDuplicates()` to merge geometrically coincident volumes/surfaces, producing genuine shared interfaces.
5. **Rebuild mapping** — Reconstruct the `tag → source-file` mapping by nearest-centroid matching, since tags are reshuffled by `removeAllDuplicates()`.
6. **Extract mesh** — When `return_internal_surfaces=False`, skip surfaces shared by more than one volume; keep only the outer surface, which is fully watertight.

---

## Output Data Format

### Using `mapping_table`

```python
# Find all shared interface faces
interface_faces = faces[mapping_table[:, 1] > 0]

# Extract all faces belonging to one volume (outer + interfaces)
vol_id = 2
vol_mask = (mapping_table[:, 0] == vol_id) | (mapping_table[:, 1] == vol_id)
vol_faces = faces[vol_mask]

# Find cone–cube interface faces
iface_1_2 = faces[(mapping_table[:, 0] == 1) & (mapping_table[:, 1] == 2)]
```

### `tet_centroids_data` format

When `return_each_tet_as_body=True`, each dict contains:

| Key | Type | Description |
|---|---|---|
| `tet_id` | `int` | Global unique tetrahedron ID |
| `centroid` | `np.ndarray` | `(3,)` centroid coordinates |
| `volume_tag` | `int` | Associated Gmsh geometric volume tag |
| `filename` | `str` | Source STEP filename |

---

## Testing

### Run the tests

```bash
# Run all tests from the repository root
pytest tests/ -v

# Run glue-related tests only
pytest tests/test_glue.py -v

# Run msh2stl-related tests only
pytest tests/test_mesh.py -k 'msh2stl' -v
```

### Examples directory resolution

Tests automatically locate a directory containing `.step` files in this order:

1. Environment variable `$STEPMESH_EXAMPLES_DIR`
2. `<repo>/examples/`
3. `<repo>/../georay/examples/`
4. Common Colab paths (`/content/georay/examples/`, etc.)

To specify manually:

```bash
STEPMESH_EXAMPLES_DIR=/path/to/examples pytest tests/ -v
```

### Test coverage

| Test file | Coverage |
|---|---|
| `tests/test_glue.py` | Outer-surface watertightness, internal interface topology, `mapping_table` consistency, `file_to_ids_dict`, parametrized mesh sizes, input subsets, `tet_as_body`, STL export |
| `tests/test_mesh.py` | `glue()` output format, `step2msh()` file writing, `vf2stl()` binary/ASCII, `msh2stl()` with/without internal faces, `msh2unv()`, error handling |

### Test dependencies

```bash
pip install pytest trimesh
```

---

## Requirements

| Dependency | Version |
|---|---|
| Python | >= 3.8 |
| `numpy` | >= 1.20.0 |
| `gmsh` | >= 4.10.0 |

System libraries: `libglu1-mesa`, `libgl1` (Linux/Colab)

---

## License

This project is licensed under the MIT License. See [LICENSE](https://github.com/gwangcode/stepmesh/blob/main/LICENSE) for details.

---

## Citation

If this tool has been useful in your research, please cite:

```bibtex
@software{stepmesh,
  author = {G. Wang},
  title = {stepmesh: A Python package for 3D STEP multi-body boolean alignment and Gmsh tetrahedral mesh processing},
  year = {2026},
  url = {https://github.com/gwangcode/stepmesh}
}
```

See [`CITATION.cff`](https://github.com/gwangcode/stepmesh/blob/main/CITATION.cff) for more details.

