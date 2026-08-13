---
title: 'stepmesh: A Python Package for STEP Multi-body Boolean Alignment and Mesh Generation'
tags:
  - Python
  - CAD
  - STEP
  - Gmsh
  - Finite Element Analysis
  - Mesh Generation
authors:
  - name: Gang Wang
    orcid: 0009-0002-4777-9376
    affiliation: 1
affiliations:
 - name: Hardin-Simmons University
   index: 1
date: 13 August 2026
bibliography: paper.bib
---

# Summary

In computer-aided engineering (CAE), computational fluid dynamics (CFD), and wave propagation physics (such as optical ray tracing and acoustic scattering), converting multi-body CAD assemblies into continuous, matching finite element meshes is a critical preprocessing step. When importing multiple independent STEP (`.step`/`.stp`) files representing separate assembly components, non-conforming interface meshes often introduce artificial gaps, node mismatches, or intersecting boundary surfaces, leading to ill-posed boundary conditions in numerical solvers.

`stepmesh` is a lightweight Python package designed to automate multi-body STEP geometry alignment, watertight surface remeshing, and conformal 3D tetrahedral mesh extraction. Powered by the Gmsh OpenCASCADE kernel [@Geuzaine2009], `stepmesh` performs Boolean fragmentation (`fragment`) and gluing (`sew`) operations on arbitrary assembly components. This process ensures perfectly matching, node-sharing interfaces between contacting bodies, generating topological watertight mesh interfaces directly in memory as structured NumPy arrays [@Harris2020] or exporting them to standard finite element formats (`.msh`, `.unv`, `.stl`).

# Statement of Need

# Statement of Need

Preparing multi-body CAD geometries for physical simulations frequently requires tedious manual interaction in commercial CAD preprocessors to ensure boundary alignment. While methods such as *CoplanarMesh* [@Wang2026CoplanarMesh] address boundary-aligned topological remeshing, ensuring strict topological watertightness and conformal (node-sharing) contact interfaces directly in memory across multi-body STEP assemblies remains a practical challenge. Furthermore, scripting low-level OpenCASCADE Boolean fragment operations in Gmsh often involves significant boilerplate code, topological entity indexing, and disk I/O overhead.

`stepmesh` addresses this gap by providing a pythonic, high-level pipeline that abstracts low-level OpenCASCADE topological operations. The key capabilities of `stepmesh` include:

1. **Conformal (Node-Sharing) Contact Interfaces**: Automatically fragments and glues multi-body assemblies, ensuring that adjacent contact interfaces share identical vertices and triangular faces without artificial gaps or overlaps.
2. **Watertight Surface & Volume Remeshing**: Guarantees topologically watertight surface boundaries and volumetric meshes, which are essential for numerical stability in finite element analysis (FEA), boundary element methods (BEM), and geometric ray-tracing solvers.
3. **In-Memory Array Extraction**: Directly extracts NumPy arrays for vertices $(N \times 3)$, surface triangle indices $(M \times 3)$, mapping tables, and element centroid coordinates, enabling seamless integration with custom Python solver pipelines.
4. **Open Ecosystem Interoperability**: Converts meshes to I-DEAS `.unv` files (compatible with the FreeCAD FEM workbench) and binary/ASCII `.stl` files.

# Core Architecture and API

`stepmesh` provides clean entry points for automated geometry processing and format conversions:

- **`glue()`**: Loads multiple STEP files, executes OpenCASCADE Boolean fragmentation, generates a 3D tetrahedral mesh in memory, and returns NumPy arrays containing vertex coordinates, surface triangle indices, mapping tables, and volumetric element centroids.
- **`step2msh()`**: Direct CAD-to-mesh pipeline converting a list of STEP file paths into a single conforming Gmsh `.msh` file with controllable global mesh size bounds (`min_size`, `max_size`).
- **`msh2unv()`, `msh2stl()`, `vf2stl()`**: High-performance format export and conversion utilities.

```python
from stepmesh import glue, step2msh, msh2unv

# Perform in-memory multi-body mesh alignment, watertight gluing, and array extraction
vertices, faces, mapping_table, file_to_ids, centroids = glue(
    step_paths=["cube.step", "cylinder.step"],
    min_size=0.5,
    max_size=2.0,
    return_each_tet_as_body=True
)

# Direct assembly conversion to an I-DEAS UNV file for FreeCAD FEM analysis
step2msh(["cube.step", "cylinder.step"], "output.msh")
msh2unv("output.msh", "output.unv")

```

# Availability and License

`stepmesh` is open-source software distributed under the MIT License. Source code, documentation, examples, and automated unit test suites are available on GitHub (https://github.com/gwangcode/stepmesh).

# References

