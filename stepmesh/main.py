import os
import sys
import gmsh
import numpy as np


def step2msh(
    step_paths: list[str],
    output_msh_path: str,
    min_size: float = 0.0,
    max_size: float = 0.0,
) -> dict[str, list[int]]:
  """Convert a list of STEP CAD files into a 3D tetrahedral finite element mesh (.msh)
  using Gmsh. Automatically performs multi-body boolean fragment operations across all imported files to ensure matching coplanar and
  contact surfaces.

  Parameters:
      step_paths (list[str]): A list of file paths to input CAD files.
        Supported formats include STEP (.step, .stp), BREP (.brep), IGES
        (.iges, .igs), and other Gmsh-compatible 3D solid formats[cite: 4]
      output_msh_path (str): The file path where the generated MSH file will be
        saved[cite: 4].
      min_size (float, optional): The minimum characteristic length control.
        Defaults to 0.0[cite: 4].
      max_size (float, optional): The maximum characteristic length control.
        Defaults to 0.0[cite: 4].

  Returns:
      dict[str, list[int]]: A dictionary mapping each STEP filename to its
      corresponding assigned physical entity IDs (1-based)[cite: 4].
  """
  gmsh.initialize()
  gmsh.option.setNumber("General.Terminal", 1)
  gmsh.model.add("multi_body_step_mesh")

  file_to_original_vols = {}
  orig_vol_to_filename = {}

  for path in step_paths:
    if not os.path.exists(path):
      print(f"Warning: STEP file not found: {path}")
      continue

    before_vols = set(gmsh.model.getEntities(dim=3))

    try:
      gmsh.merge(path)
    except Exception as e:
      raise ValueError(f"Failed to read STEP file {path}: {e}")

    gmsh.model.occ.synchronize()

    after_vols = set(gmsh.model.getEntities(dim=3))
    new_vols = [v[1] for v in (after_vols - before_vols)]

    file_name = os.path.basename(path)
    file_to_original_vols[file_name] = new_vols
    for v in new_vols:
      orig_vol_to_filename[v] = file_name

  if not file_to_original_vols:
    gmsh.finalize()
    raise ValueError(
        "Error: No valid 3D volumes detected from the provided STEP files."
    )

  all_volumes = []
  for vols in file_to_original_vols.values():
    all_volumes.extend(vols)

  input_dim_tags = [(3, v) for v in all_volumes]
  _, out_dim_tags_map = gmsh.model.occ.fragment(input_dim_tags, [])
  gmsh.model.occ.synchronize()

  new_vol_to_filename = {}
  for orig_vol, sub_entities in zip(all_volumes, out_dim_tags_map):
    fname = orig_vol_to_filename[orig_vol]
    for dim, tag in sub_entities:
      if dim == 3:
        new_vol_to_filename[tag] = fname

  final_volumes = gmsh.model.getEntities(dim=3)

  file_to_ids_dict = {filename: [] for filename in file_to_original_vols.keys()}

  for i, vol in enumerate(final_volumes):
    dim, tag = vol[0], vol[1]
    physical_tag = i + 1
    gmsh.model.addPhysicalGroup(dim, [tag], physical_tag)
    group_name = f"Domain_{physical_tag}"
    gmsh.model.setPhysicalName(dim, physical_tag, group_name)

    fname = new_vol_to_filename.get(tag)
    if fname and fname in file_to_ids_dict:
      file_to_ids_dict[fname].append(physical_tag)

  if min_size > 0:
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", min_size)
  if max_size > 0 and min_size <= max_size:
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", max_size)

  gmsh.model.mesh.generate(3)
  gmsh.write(output_msh_path)
  gmsh.finalize()

  return file_to_ids_dict


def glue(
    step_paths: list[str],
    min_size: float = 0.0,
    max_size: float = 0.0,
    return_internal_surfaces: bool = False,
    return_each_tet_as_body: bool = False,
) -> tuple:
  """Reads multiple STEP CAD files, performs multi-body boolean operations (Fragment/Gluing),

  generates a 3D tetrahedral finite element mesh in memory, extracts structured
  NumPy arrays, and establishes ownership mapping back to source files.

  This function is designed for multi-component CAD mesh generation, enabling
  conformal matching nodes across contact surfaces while tracking entity
  ownership.

  Args:
      step_paths (list[str]): List of absolute or relative file paths to the
        STEP CAD files.
      min_size (float, optional): Minimum mesh size limit
        (Mesh.CharacteristicLengthMin). Defaults to 0.0 (uses Gmsh default).
      max_size (float, optional): Maximum mesh size limit
        (Mesh.CharacteristicLengthMax). Defaults to 0.0 (uses Gmsh default).
      return_internal_surfaces (bool, optional):
        Whether to export internal shared interface meshes between contacting
        bodies.
          - False (default): Exports only the outer boundary skin mesh.
          - True: Exports mesh including all internal contact interfaces (Glue
          Interfaces).
      return_each_tet_as_body (bool, optional): Whether to treat each
        individual tetrahedral element as a distinct body for decomposition and
        mapping.
          - False (default): Groups and maps entities at the STEP file / CAD
          volume level.
          - True: Exports centroid coordinates at individual 3D tetrahedron
          granularity and populates the 5th return value.

  Returns:
      tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, list[int]], list[dict]]:
          A tuple containing 5 elements:

          1. vertices (np.ndarray):
             A float64 array of shape (N_verts, 3) representing unique 3D node
             coordinates [x, y, z].

          2. faces (np.ndarray):
             An int64 array of shape (N_faces, 3) representing triangle element
             vertex indices (0-based indexing relative to `vertices`).

          3. mapping_table (np.ndarray):
             An int32 array of shape (N_faces, 3) recording entity ownership
             and adjacency relations for each triangle face:
               - Column 0 (`pos_vol`): Positive attached Physical Volume ID (>=
                 1).
               - Column 1 (`neg_vol`): Negative attached Physical Volume ID.
                 * Equal to 0: Indicates the face is an **outer boundary
                   surface**.
                 * Greater than 0: Indicates the face is a **shared interface
                   connecting two bodies**.
               - Column 2 (`local_tri_id`): Row index of the face in the `faces`
                 array (0 to N_faces - 1).

          4. file_to_ids_dict (dict[str, list[int]]):
             A mapping dictionary where keys are STEP filenames (e.g.,
             `"cube.step"`) and values are lists of contained Physical Volume
             IDs (or tetrahedral Element IDs if `return_each_tet_as_body=True`).

          5. tet_centroids_data (list[dict]):
             A list of dictionaries containing tetrahedral element centroid
             data. Populated only when `return_each_tet_as_body=True`;
             otherwise returns an empty list `[]`. Each dictionary contains:
               - "tet_id" (int): Global unique ID of the tetrahedral element.
               - "centroid" (np.ndarray): 3D centroid coordinate vector [x, y,
                 z] of shape (3,).
               - "volume_tag" (int): Associated Gmsh geometry Volume Tag.
               - "filename" (str): Corresponding source STEP filename.

  Raises:
      ValueError: Raised if reading input STEP files fails or if no valid 3D
        volumes are detected.

  Example:
      >>> step_files = ["cone.step", "cube.step"]
      >>> verts, faces, mapping, file_map, centroids = glue(
      ...     step_paths=step_files,
      ...     min_size=1.0,
      ...     max_size=5.0,
      ...     return_internal_surfaces=True,
      ... )
      >>> print(f"Extracted Vertices: {len(verts)}, Faces: {len(faces)}")
      >>> print("File-to-ID Mapping:", file_map)
      >>> # Find all faces lying on shared interfaces:
      >>> interface_faces = mapping[mapping[:, 1] > 0]
  """
  gmsh.initialize()
  gmsh.option.setNumber("General.Terminal", 0)
  gmsh.model.add("multi_body_step_mesh")

  if min_size > 0:
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", min_size)
  if max_size > 0 and min_size <= max_size:
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", max_size)

  file_to_original_vols = {}
  orig_vol_to_filename = {}

  for path in step_paths:
    if not os.path.exists(path):
      continue
    before_vols = set(gmsh.model.getEntities(dim=3))
    try:
      gmsh.merge(path)
    except Exception as e:
      raise ValueError(f"Failed to read STEP file {path}: {e}")
    gmsh.model.occ.synchronize()
    after_vols = set(gmsh.model.getEntities(dim=3))
    new_vols = [v[1] for v in (after_vols - before_vols)]

    fname = os.path.basename(path)
    file_to_original_vols[fname] = new_vols
    for v in new_vols:
      orig_vol_to_filename[v] = fname

  all_volumes = []
  for vols in file_to_original_vols.values():
    all_volumes.extend(vols)

  new_vol_to_filename = {}
  if all_volumes:
    input_dim_tags = [(3, v) for v in all_volumes]
    _, out_dim_tags_map = gmsh.model.occ.fragment(input_dim_tags, [])
    gmsh.model.occ.synchronize()

    for orig_vol, sub_entities in zip(all_volumes, out_dim_tags_map):
      fname = orig_vol_to_filename[orig_vol]
      for dim, tag in sub_entities:
        if dim == 3:
          new_vol_to_filename[tag] = fname

  geom_volumes = gmsh.model.getEntities(dim=3)

  vol_tag_to_physical_id = {}
  vol_tag_to_filename = {}

  for i, vol in enumerate(geom_volumes):
    dim, tag = vol[0], vol[1]
    physical_tag = i + 1
    gmsh.model.addPhysicalGroup(dim, [tag], physical_tag)
    gmsh.model.setPhysicalName(dim, physical_tag, f"Domain_{physical_tag}")
    vol_tag_to_physical_id[tag] = physical_tag
    vol_tag_to_filename[tag] = new_vol_to_filename.get(tag, "unknown")

  gmsh.model.mesh.generate(3)

  node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
  coord_array = np.zeros((int(np.max(node_tags)) + 1, 3), dtype=np.float64)
  coord_array[node_tags] = node_coords.reshape(-1, 3)

  file_to_ids_dict = {fname: [] for fname in file_to_original_vols.keys()}
  tet_centroids_data = []

  if return_each_tet_as_body:
    elem_types_3d, elem_tags_3d, elem_node_tags_3d = (
        gmsh.model.mesh.getElements(3)
    )

    face_to_vols = {}
    for etype, tags, node_tags_list in zip(
        elem_types_3d, elem_tags_3d, elem_node_tags_3d
    ):
      if etype == 4:
        nodes_matrix = node_tags_list.reshape(-1, 4)
        for tet_tag, tet_nodes in zip(tags, nodes_matrix):
          tet_id = int(tet_tag)

          p0 = coord_array[tet_nodes[0]]
          p1 = coord_array[tet_nodes[1]]
          p2 = coord_array[tet_nodes[2]]
          p3 = coord_array[tet_nodes[3]]
          centroid = (p0 + p1 + p2 + p3) / 4.0

          _, _, _, vol_geom_tag = gmsh.model.mesh.getElement(tet_tag)
          vol_phys_id = vol_tag_to_physical_id.get(vol_geom_tag, 1)

          faces_of_tet = [
              tuple(sorted((tet_nodes[0], tet_nodes[1], tet_nodes[2]))),
              tuple(sorted((tet_nodes[0], tet_nodes[3], tet_nodes[1]))),
              tuple(sorted((tet_nodes[0], tet_nodes[2], tet_nodes[3]))),
              tuple(sorted((tet_nodes[1], tet_nodes[3], tet_nodes[2]))),
          ]
          for f_nodes in faces_of_tet:
            if f_nodes not in face_to_vols:
              face_to_vols[f_nodes] = []
            if vol_phys_id not in face_to_vols[f_nodes]:
              face_to_vols[f_nodes].append(vol_phys_id)

          source_file = vol_tag_to_filename.get(vol_geom_tag, "unknown")

          if source_file in file_to_ids_dict:
            file_to_ids_dict[source_file].append(tet_id)

          tet_centroids_data.append({
              "tet_id": tet_id,
              "centroid": centroid,
              "volume_tag": int(vol_geom_tag),
              "filename": source_file,
          })

    raw_faces = []
    mapping_table = []
    for local_tri_id, (f_nodes, vols) in enumerate(face_to_vols.items()):
      raw_faces.append(np.array(f_nodes, dtype=np.int64))
      pos_vol = vols[0]
      neg_vol = vols[1] if len(vols) > 1 else 0
      mapping_table.append([pos_vol, neg_vol, local_tri_id])

    raw_faces = np.array(raw_faces, dtype=np.int64)
    mapping_table = np.array(mapping_table, dtype=np.int32)

  else:
    for vol_geom_tag, phys_id in vol_tag_to_physical_id.items():
      fname = vol_tag_to_filename.get(vol_geom_tag)
      if fname and fname in file_to_ids_dict:
        file_to_ids_dict[fname].append(phys_id)

    if return_internal_surfaces:
      elem_types_3d, elem_tags_3d, elem_node_tags_3d = (
          gmsh.model.mesh.getElements(3)
      )
      face_to_vols = {}
      for etype, tags, node_tags_list in zip(
          elem_types_3d, elem_tags_3d, elem_node_tags_3d
      ):
        if etype == 4:
          nodes_matrix = node_tags_list.reshape(-1, 4)
          for tet_tag, tet_nodes in zip(tags, nodes_matrix):
            _, _, _, vol_geom_tag = gmsh.model.mesh.getElement(tet_tag)
            vol_phys_id = vol_tag_to_physical_id.get(vol_geom_tag, 1)

            faces_of_tet = [
                tuple(sorted((tet_nodes[0], tet_nodes[1], tet_nodes[2]))),
                tuple(sorted((tet_nodes[0], tet_nodes[3], tet_nodes[1]))),
                tuple(sorted((tet_nodes[0], tet_nodes[2], tet_nodes[3]))),
                tuple(sorted((tet_nodes[1], tet_nodes[3], tet_nodes[2]))),
            ]
            for f_nodes in faces_of_tet:
              if f_nodes not in face_to_vols:
                face_to_vols[f_nodes] = []
              if vol_phys_id not in face_to_vols[f_nodes]:
                face_to_vols[f_nodes].append(vol_phys_id)

      raw_faces = []
      mapping_table = []
      for local_tri_id, (f_nodes, vols) in enumerate(face_to_vols.items()):
        raw_faces.append(np.array(f_nodes, dtype=np.int64))
        pos_vol = vols[0]
        neg_vol = vols[1] if len(vols) > 1 else 0
        mapping_table.append([pos_vol, neg_vol, local_tri_id])

      raw_faces = np.array(raw_faces, dtype=np.int64)
      mapping_table = np.array(mapping_table, dtype=np.int32)
    else:
      surf_ownership = {}
      for vol in geom_volumes:
        vol_geom_tag = vol[1]
        vol_phys_id = vol_tag_to_physical_id[vol_geom_tag]

        boundaries = gmsh.model.getBoundary(
            [vol], combined=False, oriented=True
        )
        for b in boundaries:
          if b[0] == 2:
            signed_surf_tag = int(b[1])
            surf_tag = abs(signed_surf_tag)
            is_reversed = signed_surf_tag < 0

            if surf_tag not in surf_ownership:
              surf_ownership[surf_tag] = []
            surf_ownership[surf_tag].append((vol_phys_id, is_reversed))

      raw_faces = []
      mapping_table = []
      local_tri_id = 0

      for surf_tag, owners in surf_ownership.items():
        elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(
            2, surf_tag
        )

        for i, etype in enumerate(elem_types):
          if etype == 2:
            t_tags = elem_tags[i]
            n_tags = elem_node_tags[i].reshape(-1, 3)

            for j, _ in enumerate(t_tags):
              nodes = n_tags[j]
              vol_id, is_reversed = owners[0]

              if is_reversed:
                face_nodes = np.array(
                    [nodes[0], nodes[2], nodes[1]], dtype=np.int64
                )
              else:
                face_nodes = np.array(
                    [nodes[0], nodes[1], nodes[2]], dtype=np.int64
                )

              pos_vol = vol_id
              neg_vol = owners[1][0] if len(owners) > 1 else 0

              raw_faces.append(face_nodes)
              mapping_table.append([pos_vol, neg_vol, local_tri_id])
              local_tri_id += 1

      raw_faces = np.array(raw_faces, dtype=np.int64)
      mapping_table = np.array(mapping_table, dtype=np.int32)

  unique_node_tags = np.unique(raw_faces)
  vertices = coord_array[unique_node_tags]

  tag_to_idx = {tag: idx for idx, tag in enumerate(unique_node_tags)}
  faces = np.vectorize(tag_to_idx.get)(raw_faces)

  gmsh.finalize()

  return vertices, faces, mapping_table, file_to_ids_dict, tet_centroids_data


def vf2stl(
    vertices: np.ndarray, faces: np.ndarray, output_stl_path: str, binary: bool = True
):
  """Export input vertices and faces NumPy arrays to a standard STL file."""
  tri_verts = vertices[faces]

  v0 = tri_verts[:, 0, :]
  v1 = tri_verts[:, 1, :]
  v2 = tri_verts[:, 2, :]

  edge1 = v1 - v0
  edge2 = v2 - v0

  normals = np.cross(edge1, edge2)

  norms = np.linalg.norm(normals, axis=1, keepdims=True)
  norms[norms == 0] = 1e-12
  normals = normals / norms

  n_faces = len(faces)

  if binary:
    header = b"Exported by NumPy & Gmsh Workflow"
    header = header.ljust(80, b"\0")

    dtype = np.dtype([
        ("normal", "<f4", (3,)),
        ("vertices", "<f4", (3, 3)),
        ("attr", "<u2"),
    ])

    facet_array = np.zeros(n_faces, dtype=dtype)
    facet_array["normal"] = normals
    facet_array["vertices"] = tri_verts
    facet_array["attr"] = 0

    with open(output_stl_path, "wb") as f:
      f.write(header)
      f.write(np.uint32(n_faces).tobytes())
      f.write(facet_array.tobytes())

  else:
    with open(output_stl_path, "w", encoding="utf-8") as f:
      f.write("solid exported_mesh\n")
      for i in range(n_faces):
        nx, ny, nz = normals[i]
        p0, p1, p2 = tri_verts[i]

        f.write(f"  facet normal {nx:.6e} {ny:.6e} {nz:.6e}\n")
        f.write("    outer loop\n")
        f.write(f"      vertex {p0[0]:.6e} {p0[1]:.6e} {p0[2]:.6e}\n")
        f.write(f"      vertex {p1[0]:.6e} {p1[1]:.6e} {p1[2]:.6e}\n")
        f.write(f"      vertex {p2[0]:.6e} {p2[1]:.6e} {p2[2]:.6e}\n")
        f.write("    endloop\n")
        f.write("  endfacet\n")
      f.write("endsolid exported_mesh\n")


def msh2unv(msh_path: str, unv_path: str) -> str:
  """Convert a Gmsh .msh file to an I-deas .unv file for FreeCAD FEM workbench."""
  if not os.path.exists(msh_path):
    raise FileNotFoundError(f"Input MSH file not found: {msh_path}")

  is_init = gmsh.is_initialized()
  if not is_init:
    gmsh.initialize()

  try:
    gmsh.clear()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.open(msh_path)
    gmsh.option.setNumber("Mesh.SaveAll", 1)

    output_dir = os.path.dirname(os.path.abspath(unv_path))
    if output_dir and not os.path.exists(output_dir):
      os.makedirs(output_dir)

    gmsh.write(unv_path)

  except Exception as e:
    raise RuntimeError(f"Failed to convert MSH to UNV: {e}")

  finally:
    if not is_init:
      gmsh.finalize()

  return os.path.abspath(unv_path)


def msh2stl(msh_path: str, stl_path: str) -> str:
  if not os.path.exists(msh_path):
    raise FileNotFoundError(f"Input MSH file not found: {msh_path}")

  is_init = gmsh.is_initialized()
  if not is_init:
    gmsh.initialize()

  try:
    gmsh.clear()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.open(msh_path)

    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    node_map = {tag: i for i, tag in enumerate(node_tags)}
    nodes = node_coords.reshape(-1, 3)

    elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(dim=3)

    tets = []
    for e_type, e_nodes in zip(elem_types, elem_node_tags):
      if e_type == 4:
        tets = e_nodes.reshape(-1, 4)
        break

    if len(tets) == 0:
      raise RuntimeError("No 3D tetrahedral elements found in the mesh file.")

    face_count = {}

    for tet in tets:
      p = [node_map[tag] for tag in tet]
      sub_faces = [
          (p[0], p[2], p[1]),
          (p[0], p[1], p[3]),
          (p[1], p[2], p[3]),
          (p[2], p[0], p[3]),
      ]
      for f in sub_faces:
        sorted_f = tuple(sorted(f))
        if sorted_f not in face_count:
          face_count[sorted_f] = []
        face_count[sorted_f].append(f)

    boundary_triangles = []
    for sorted_f, f_list in face_count.items():
      if len(f_list) == 1:
        boundary_triangles.append(f_list[0])

    output_dir = os.path.dirname(os.path.abspath(stl_path))
    if output_dir and not os.path.exists(output_dir):
      os.makedirs(output_dir)

    with open(stl_path, "w") as f:
      f.write("solid GmshModel\n")
      for tri in boundary_triangles:
        pt0 = nodes[tri[0]]
        pt1 = nodes[tri[1]]
        pt2 = nodes[tri[2]]

        f.write("  facet normal 0.0 0.0 0.0\n")
        f.write("    outer loop\n")
        f.write(f"      vertex {pt0[0]} {pt0[1]} {pt0[2]}\n")
        f.write(f"      vertex {pt1[0]} {pt1[1]} {pt1[2]}\n")
        f.write(f"      vertex {pt2[0]} {pt2[1]} {pt2[2]}\n")
        f.write("    endloop\n")
        f.write("  endfacet\n")
      f.write("endsolid GmshModel\n")

    if not (os.path.exists(stl_path) and os.path.getsize(stl_path) > 0):
      raise RuntimeError("Generated STL file is empty.")

  except Exception as e:
    raise RuntimeError(f"Failed to export STL: {e}")

  finally:
    if not is_init:
      gmsh.finalize()

  return os.path.abspath(stl_path)
