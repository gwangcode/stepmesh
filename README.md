# stepmesh 

[![DOI](https://zenodo.org/badge/1332405493.svg)](https://doi.org/10.5281/zenodo.21909276)

A Python package for processing 3D STEP/CAD files, performing multi-body boolean operations, and generating tetrahedral meshes (`.msh`), STL files (`.stl`), and UNV files (`.unv`) using **Gmsh** and **NumPy**.

## Features

- **Multi-body STEP Mesh Generation**: Automatically fragment and glue multi-body STEP files to ensure continuous and matching contact interfaces.
- **Mesh Data Extraction**: Extract NumPy arrays for vertices, triangle faces, mapping tables, and tetrahedron centroids directly in memory.
- **Format Conversion**: Convert Gmsh `.msh` files to `.unv` (compatible with FreeCAD FEM workbench) and export to STL formats.
- **STEP/CAD integration** — Convert multi-body STEP assemblies remeshed by [`stepmesh`](https://github.com/gwangcode/stepmesh) via a lightweight adapter (`georay.stepmesh_adapter`).

## Installation

You can install `stepmesh` directly from GitHub using `pip`:

```bash
pip install git+https://github.com/gwangcode/stepmesh.git

```

## Quick Start

```python
from stepmesh import glue, step2msh, msh2stl, msh2unv

# 1. Mesh STEP files and extract NumPy arrays in memory
vertices, faces, mapping_table, file_to_ids_dict = glue(
    step_paths=["cube.step", "cylinder.step"],
    min_size=0.5,
    max_size=2.0,
    return_internal_surfaces=True
)

print(f"Extracted {len(vertices)} vertices and {len(faces)} faces.")

# 2. Convert STEP files directly to a 3D .msh file
file_to_ids = step2msh(
    step_paths=["cube.step", "cylinder.step"],
    output_msh_path="output.msh",
    min_size=0.5,
    max_size=2.0
)

# 3. Format Conversions
msh2stl("output.msh", "output.stl")
msh2unv("output.msh", "output.unv")

```

## Functions API

* **`step2msh(step_paths, output_msh_path, min_size=0.0, max_size=0.0)`**: Converts a list of STEP CAD files into a 3D tetrahedral finite element mesh (`.msh`).
* **`glue(step_paths, min_size=0.0, max_size=0.0, return_internal_surfaces=False, return_each_tet_as_body=False)`**: Generates a 3D tetrahedral mesh in memory and returns NumPy arrays for vertices, faces, mapping tables, and entity IDs.
* **`vf2stl(vertices, faces, output_stl_path, binary=True)`**: Exports vertex and face arrays directly to a binary or ASCII STL file.
* **`msh2unv(msh_path, unv_path)`**: Converts a Gmsh `.msh` file to an I-deas `.unv` file (ready for FreeCAD FEM).
* **`msh2stl(msh_path, stl_path)`**: Extracts outer boundary surface triangles from a `.msh` file and exports them to an ASCII STL file.

## Requirements

* Python >= 3.8
* `gmsh`
* `numpy`

## License

This project is licensed under the MIT License - see the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.

