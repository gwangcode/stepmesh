import os, sys, gmsh, numpy as np


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

  file_to_volumes = {}

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
    new_vols = list(after_vols - before_vols)

    file_name = os.path.basename(path)
    file_to_volumes[file_name] = new_vols

  if not file_to_volumes:
    gmsh.finalize()
    raise ValueError("Error: No valid 3D volumes detected from the provided STEP files.")

  all_volumes = []
  for vols in file_to_volumes.values():
    all_volumes.extend(vols)

  out_dim_tags, _ = gmsh.model.occ.fragment(all_volumes, [])
  gmsh.model.occ.synchronize()

  final_volumes = gmsh.model.getEntities(dim=3)
  n = len(final_volumes)

  file_to_ids_dict = {filename: [] for filename in file_to_volumes.keys()}

  for i, vol in enumerate(final_volumes):
    dim, tag = vol[0], vol[1]
    physical_tag = i + 1
    gmsh.model.addPhysicalGroup(dim, [tag], physical_tag)
    group_name = f"Domain_{physical_tag}"
    gmsh.model.setPhysicalName(dim, physical_tag, group_name)

  current_idx = 1
  for filename, original_vols in file_to_volumes.items():
    count = max(1, len(original_vols))
    assigned_ids = list(range(current_idx, current_idx + count))
    file_to_ids_dict[filename] = assigned_ids
    current_idx += count

  if current_idx - 1 < n:
    for idx in range(current_idx, n + 1):
      last_key = list(file_to_ids_dict.keys())[-1]
      if idx not in file_to_ids_dict[last_key]:
        file_to_ids_dict[last_key].append(idx)

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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, list[int]], list[dict]]:
  """Read a list of STEP files, execute multi-body boolean gluing across all,
  generate a 3D tetrahedral mesh in memory, extract structured NumPy arrays,
  and return a mapping dictionary linking each STEP filename to its entity IDs.
  If return_each_tet_as_body is True, also calculates and returns individual
  tetrahedron centroids, maps each tet to its originating STEP file, and
  includes all internal/external tetrahedron faces in the faces array.

  Parameters:
      step_paths (list[str]): A list of file paths to input CAD files.
      min_size (float, optional): Minimum characteristic length control.
      max_size (float, optional): Maximum characteristic length control.
      return_internal_surfaces (bool, optional): If True, also extract internal
        surfaces shared between tetrahedra.
      return_each_tet_as_body (bool, optional): If True, treats each individual
        tetrahedron as an independent body and maps them accordingly.

  Returns:
      tuple:
          - vertices (np.ndarray): 3D coordinates of extracted vertices.
          - faces (np.ndarray): Triangle node indices (includes internal faces if return_each_tet_as_body=True).
          - mapping_table (np.ndarray): Entity contact and ownership mapping table.
          - file_to_ids_dict (dict[str, list[int]]): Mapping from filename to entity/tetrahedron IDs.
          - tet_centroids_data (list[dict]): List containing tet IDs, centroids, and source file names.
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
    new_vols = list(after_vols - before_vols)

    fname = os.path.basename(path)
    file_to_original_vols[fname] = new_vols
    for v in new_vols:
      orig_vol_to_filename[v] = fname

  all_volumes = []
  for vols in file_to_original_vols.values():
    all_volumes.extend(vols)

  if all_volumes:
    gmsh.model.occ.fragment(all_volumes, [])
    gmsh.model.occ.synchronize()

  geom_volumes = gmsh.model.getEntities(dim=3)
  n = len(geom_volumes)

  vol_tag_to_physical_id = {}
  for i, vol in enumerate(geom_volumes):
    dim, tag = vol[0], vol[1]
    physical_tag = i + 1
    gmsh.model.addPhysicalGroup(dim, [tag], physical_tag)
    gmsh.model.setPhysicalName(dim, physical_tag, f"Domain_{physical_tag}")
    vol_tag_to_physical_id[tag] = physical_tag

  gmsh.model.mesh.generate(3)

  # 获取所有节点坐标用于计算中心和顶点
  node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
  coord_array = np.zeros((int(np.max(node_tags)) + 1, 3), dtype=np.float64)
  coord_array[node_tags] = node_coords.reshape(-1, 3)

  file_to_ids_dict = {fname: [] for fname in file_to_original_vols.keys()}
  tet_centroids_data = []

  # 如果开启了每个四面体作为一个独立的 body
  if return_each_tet_as_body:
    elem_types_3d, elem_tags_3d, elem_node_tags_3d = gmsh.model.mesh.getElements(3)

    face_to_vols = {}
    for etype, tags, node_tags_list in zip(elem_types_3d, elem_tags_3d, elem_node_tags_3d):
      if etype == 4:  # 4节点四面体
        nodes_matrix = node_tags_list.reshape(-1, 4)
        for tet_tag, tet_nodes in zip(tags, nodes_matrix):
          tet_id = int(tet_tag)

          # 计算四面体中心坐标
          p0 = coord_array[tet_nodes[0]]
          p1 = coord_array[tet_nodes[1]]
          p2 = coord_array[tet_nodes[2]]
          p3 = coord_array[tet_nodes[3]]
          centroid = (p0 + p1 + p2 + p3) / 4.0

          # 获取该四面体所属的几何实体 (Volume)
          _, _, _, vol_geom_tag = gmsh.model.mesh.getElement(tet_tag)
          vol_phys_id = vol_tag_to_physical_id.get(vol_geom_tag, 1)

          # 收集四面体的 4 个面以便后续构建包含内部面的 faces
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

          # 溯源原始文件名
          source_file = "unknown"
          try:
            ancestors = gmsh.model.occ.getAncestor(3, vol_geom_tag)
            if ancestors:
              for anc in ancestors:
                if anc in orig_vol_to_filename:
                  source_file = orig_vol_to_filename[anc]
                  break
          except Exception:
            pass

          if source_file == "unknown" and file_to_original_vols:
            source_file = list(file_to_original_vols.keys())[0]

          if source_file in file_to_ids_dict:
            file_to_ids_dict[source_file].append(tet_id)
          else:
            file_to_ids_dict[source_file] = [tet_id]

          tet_centroids_data.append({
            "tet_id": tet_id,
            "centroid": centroid,
            "volume_tag": int(vol_geom_tag),
            "filename": source_file
          })

    # 当 return_each_tet_as_body=True 时，直接由四面体推导所有内/外表面
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
    # 未开启时执行原有的表面提取或 internal_surfaces 逻辑
    current_idx = 1
    for filename, original_vols in file_to_original_vols.items():
      count = max(1, len(original_vols))
      assigned_ids = list(range(current_idx, current_idx + count))
      file_to_ids_dict[filename] = assigned_ids
      current_idx += count

    for idx in range(current_idx, n + 1):
      last_key = list(file_to_ids_dict.keys())[-1]
      if idx not in file_to_ids_dict[last_key]:
        file_to_ids_dict[last_key].append(idx)

    if return_internal_surfaces:
      elem_types_3d, elem_tags_3d, elem_node_tags_3d = gmsh.model.mesh.getElements(3)
      face_to_vols = {}
      for etype, tags, node_tags_list in zip(elem_types_3d, elem_tags_3d, elem_node_tags_3d):
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

        boundaries = gmsh.model.getBoundary([vol], combined=False, oriented=True)
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
        elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(2, surf_tag)

        for i, etype in enumerate(elem_types):
          if etype == 2:
            t_tags = elem_tags[i]
            n_tags = elem_node_tags[i].reshape(-1, 3)

            for j, _ in enumerate(t_tags):
              nodes = n_tags[j]
              vol_id, is_reversed = owners[0]

              if is_reversed:
                face_nodes = np.array([nodes[0], nodes[2], nodes[1]], dtype=np.int64)
              else:
                face_nodes = np.array([nodes[0], nodes[1], nodes[2]], dtype=np.int64)

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

  if return_each_tet_as_body:
    return vertices, faces, mapping_table, file_to_ids_dict, tet_centroids_data

  return vertices, faces, mapping_table, file_to_ids_dict


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

    dtype = np.dtype([("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attr", "<u2")])

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
  """
  Convert a Gmsh .msh file to an I-deas .unv file for FreeCAD FEM workbench.
  """
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


import os
import gmsh
import numpy as np


def msh2stl(msh_path: str, stl_path: str) -> str:
  """
  直接从 .msh 文件的四面体网格中提取外部包络三角形面，并写入标准的 STL 文件。
  """
  if not os.path.exists(msh_path):
    raise FileNotFoundError(f"Input MSH file not found: {msh_path}")

  is_init = gmsh.is_initialized()
  if not is_init:
    gmsh.initialize()

  try:
    gmsh.clear()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.open(msh_path)

    # 1. 获取所有节点坐标
    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    node_map = {tag: i for i, tag in enumerate(node_tags)}
    nodes = node_coords.reshape(-1, 3)

    # 2. 获取所有 3D 四面体单元 (Element type 4 代表 4节点四面体)
    elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(dim=3)

    tets = []
    for e_type, e_nodes in zip(elem_types, elem_node_tags):
      if e_type == 4:  # 4-node tetrahedron
        tets = e_nodes.reshape(-1, 4)
        break

    if len(tets) == 0:
      raise RuntimeError("No 3D tetrahedral elements found in the mesh file.")

    # 3. 提取所有四面体的 4 个边界三角形面，并统计出现次数
    # 内部共享面会出现 2 次，而只出现 1 次的面就是外部边界
    face_count = {}

    for tet in tets:
      # 映射为 0-based 索引
      p = [node_map[tag] for tag in tet]
      # 四面体的 4 个面组合（注意保持一致的顶点顺时针/逆时针走向）
      sub_faces = [
        (p[0], p[2], p[1]),
        (p[0], p[1], p[3]),
        (p[1], p[2], p[3]),
        (p[2], p[0], p[3])
      ]
      for f in sub_faces:
        # 对顶点进行排序组合作为无向边的 key，或者直接记录定向面
        # 为了统计边界，使用有序化小技巧：对内部面正反各出现一次
        sorted_f = tuple(sorted(f))
        if sorted_f not in face_count:
          face_count[sorted_f] = []
        face_count[sorted_f].append(f)

    # 筛选出只出现一次的面（即外表面）
    boundary_triangles = []
    for sorted_f, f_list in face_count.items():
      if len(f_list) == 1:
        boundary_triangles.append(f_list[0])

    output_dir = os.path.dirname(os.path.abspath(stl_path))
    if output_dir and not os.path.exists(output_dir):
      os.makedirs(output_dir)

    # 4. 手动写入标准的 ASCII STL 文件
    with open(stl_path, 'w') as f:
      f.write("solid GmshModel\n")
      for tri in boundary_triangles:
        pt0 = nodes[tri[0]]
        pt1 = nodes[tri[1]]
        pt2 = nodes[tri[2]]

        # 简单计算法向量（可选，填 0 也可以，这里写 0 释放通用兼容性）
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