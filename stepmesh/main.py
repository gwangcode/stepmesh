"""stepmesh.main
================

STEP CAD → 多体 conformal 有限元网格生成。

主要函数：
    step2msh()  写入 .msh 文件
    glue()      在内存中返回 (vertices, faces, mapping_table, file_to_ids_dict,
                tet_centroids_data)
    vf2stl()    顶点/面数组 → STL
    msh2unv()   .msh → I-deas .unv
    msh2stl()   .msh → STL（仅外表面）

多体网格生成的关键步骤（修复版）：

1. 每个 STEP 用 ``gmsh.model.occ.importShapes()`` 导入，走 OCC 拓扑路径。
   （``gmsh.merge()`` 走 legacy 路径，后续 ``removeAllDuplicates()`` 无效。）
2. 用 ``fragment([first], rest)`` 切开互相接触的实体。
   （``fragment(all, [])`` 在 Gmsh 中不会切开接触面。）
3. 记录 fragment 后每个 volume 的质心。
4. ``removeAllDuplicates()`` 合并几何上完全重合的 volume / surface，
   得到真正的 shared interface（一份几何两份引用）。
5. 用质心最近邻重建 tag → source-file 映射（removeDup 后 tag 已重排）。
6. ``return_internal_surfaces=False`` 时，跳过被多个 volume 共享的接口面，
   只保留外表面，结果完全 watertight。
"""

import os
import sys

import gmsh
import numpy as np


# ---------------------------------------------------------------------------
# 内部工具
# ---------------------------------------------------------------------------

def _record_volume_centroids():
    """记录当前模型里所有 volume 的质心。"""
    centroids = {}
    for _, tag in gmsh.model.getEntities(dim=3):
        try:
            c = gmsh.model.occ.getCenterOfMass(3, tag)
            centroids[tag] = np.array(c, dtype=np.float64)
        except Exception:
            bb = gmsh.model.occ.getBoundingBox(3, tag)
            centroids[tag] = np.array([
                (bb[0] + bb[3]) / 2.0,
                (bb[1] + bb[4]) / 2.0,
                (bb[2] + bb[5]) / 2.0,
            ], dtype=np.float64)
    return centroids


def _rebuild_volume_to_filename(old_vol_to_filename, old_centroids):
    """removeAllDuplicates 之后，用质心最近邻把旧 tag→filename 映射到新 tag。"""
    new_map = {}
    for _, tag in gmsh.model.getEntities(dim=3):
        try:
            c = gmsh.model.occ.getCenterOfMass(3, tag)
            com = np.array(c, dtype=np.float64)
        except Exception:
            bb = gmsh.model.occ.getBoundingBox(3, tag)
            com = np.array([
                (bb[0] + bb[3]) / 2.0,
                (bb[1] + bb[4]) / 2.0,
                (bb[2] + bb[5]) / 2.0,
            ], dtype=np.float64)

        best_tag, best_dist = None, float("inf")
        for old_tag, old_com in old_centroids.items():
            d = float(np.sum((com - old_com) ** 2))
            if d < best_dist:
                best_dist, best_tag = d, old_tag

        if best_tag is not None and best_tag in old_vol_to_filename:
            new_map[tag] = old_vol_to_filename[best_tag]
        else:
            new_map[tag] = "unknown"
    return new_map


def _fragment_all_volumes(all_volumes):
    """对多个实体做 fragment，正确切开互相接触的实体。

    必须用 ``fragment([first], rest)`` 形式。若用 ``fragment(all, [])``，
    Gmsh 不会切开互相接触的实体，后续 removeAllDuplicates 也没有东西可合并。

    Returns:
        dict[int, int]: frag volume tag -> original volume tag
    """
    result = {}
    if len(all_volumes) == 0:
        return result
    if len(all_volumes) == 1:
        result[all_volumes[0]] = all_volumes[0]
        return result

    objects = [(3, all_volumes[0])]
    tools = [(3, v) for v in all_volumes[1:]]
    _, out_map = gmsh.model.occ.fragment(objects, tools)
    gmsh.model.occ.synchronize()

    for i, sub in enumerate(out_map):
        orig = all_volumes[i]
        for dim, tag in sub:
            if dim == 3:
                result[tag] = orig
    return result


def _import_and_index_step_files(step_paths):
    """按顺序导入 STEP，返回每个文件的原始 volume tag。

    Returns:
        (file_to_original_vols, orig_vol_to_filename)
    """
    file_to_original_vols = {}
    orig_vol_to_filename = {}

    for path in step_paths:
        if not os.path.exists(path):
            print(f"Warning: STEP file not found: {path}")
            continue

        before_vols = set(gmsh.model.getEntities(dim=3))
        try:
            gmsh.model.occ.importShapes(path)
        except Exception as e:
            raise ValueError(f"Failed to read STEP file {path}: {e}")
        gmsh.model.occ.synchronize()
        after_vols = set(gmsh.model.getEntities(dim=3))
        new_vols = [v[1] for v in (after_vols - before_vols)]

        fname = os.path.basename(path)
        file_to_original_vols[fname] = new_vols
        for v in new_vols:
            orig_vol_to_filename[v] = fname

    return file_to_original_vols, orig_vol_to_filename


# ---------------------------------------------------------------------------
# step2msh
# ---------------------------------------------------------------------------

def step2msh(
    step_paths: list[str],
    output_msh_path: str,
    min_size: float = 0.0,
    max_size: float = 0.0,
) -> dict[str, list[int]]:
    """Convert a list of STEP CAD files into a 3D tetrahedral finite element
    mesh (.msh) using Gmsh.

    Performs multi-body boolean fragment operations across all imported files
    to ensure matching coplanar and contact surfaces, then merges geometrically
    coincident volumes/surfaces into shared interfaces. Each original CAD
    body becomes one physical volume group.

    Parameters:
        step_paths (list[str]): Paths to input CAD files. STEP (.step/.stp),
            BREP (.brep), IGES (.iges/.igs) and other OCC-supported formats.
        output_msh_path (str): Output .msh file path.
        min_size (float, optional): Mesh.CharacteristicLengthMin. 0 = Gmsh
            default.
        max_size (float, optional): Mesh.CharacteristicLengthMax. 0 = Gmsh
            default.

    Returns:
        dict[str, list[int]]: STEP filename -> assigned physical volume IDs
        (1-based).
    """
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.model.add("multi_body_step_mesh")

    file_to_original_vols, orig_vol_to_filename = _import_and_index_step_files(
        step_paths
    )

    if not file_to_original_vols:
        gmsh.finalize()
        raise ValueError(
            "Error: No valid 3D volumes detected from the provided STEP files."
        )

    all_volumes = []
    for vols in file_to_original_vols.values():
        all_volumes.extend(vols)

    # --- fragment: 切开所有互相接触的实体 ---
    frag_vol_to_orig = _fragment_all_volumes(all_volumes)
    new_vol_to_filename = {
        t: orig_vol_to_filename[o] for t, o in frag_vol_to_orig.items()
    }

    # --- 记录质心（removeDup 前）---
    frag_vol_centroids = _record_volume_centroids()

    # --- 合并几何上完全重合的 volume / surface ---
    gmsh.model.occ.removeAllDuplicates()
    gmsh.model.occ.synchronize()

    # --- 用质心重建 tag → filename（tag 已被重排）---
    new_vol_to_filename = _rebuild_volume_to_filename(
        new_vol_to_filename, frag_vol_centroids
    )

    final_volumes = gmsh.model.getEntities(dim=3)
    file_to_ids_dict = {fname: [] for fname in file_to_original_vols.keys()}

    for i, vol in enumerate(final_volumes):
        dim, tag = vol[0], vol[1]
        physical_tag = i + 1
        gmsh.model.addPhysicalGroup(dim, [tag], physical_tag)
        gmsh.model.setPhysicalName(dim, physical_tag, f"Domain_{physical_tag}")

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


# ---------------------------------------------------------------------------
# glue
# ---------------------------------------------------------------------------

def glue(
    step_paths: list[str],
    min_size: float = 0.0,
    max_size: float = 0.0,
    return_internal_surfaces: bool = False,
    return_each_tet_as_body: bool = False,
) -> tuple:
    """Reads multiple STEP CAD files, performs multi-body boolean operations
    (Fragment + removeAllDuplicates), generates a 3D tetrahedral finite element
    mesh in memory, extracts structured NumPy arrays, and establishes ownership
    mapping back to source files.

    This function is designed for multi-component CAD mesh generation, enabling
    conformal matching nodes across contact surfaces while tracking entity
    ownership.

    Args:
        step_paths (list[str]): List of STEP file paths.
        min_size (float, optional): Mesh.CharacteristicLengthMin.
            Defaults to 0.0 (Gmsh default).
        max_size (float, optional): Mesh.CharacteristicLengthMax.
            Defaults to 0.0 (Gmsh default).
        return_internal_surfaces (bool, optional):
            Whether to include internal contact interfaces.

            - False (default): Export only the outer boundary skin mesh,
              guaranteed watertight (each edge shared by exactly 2 triangles).
            - True: Export mesh including all internal contact interfaces.
              The interface between two bodies is represented by a single
              shared geometric surface referenced by both volumes; every
              interface edge is then shared by 3 triangles
              (2 outer + 1 interface), which is the expected topology for a
              conformal multi-domain mesh.
        return_each_tet_as_body (bool, optional):
            - False (default): Group and map entities at STEP file / CAD
              volume level.
            - True: Emit per-tetrahedron centroid data in the 5th return
              value.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, list[int]], list[dict]]:

        1. vertices (np.ndarray):
            float64 array of shape (N_verts, 3), unique 3D node coordinates.

        2. faces (np.ndarray):
            int64 array of shape (N_faces, 3), triangle vertex indices
            (0-based, relative to ``vertices``).

        3. mapping_table (np.ndarray):
            int32 array of shape (N_faces, 3):

            - Column 0 (``pos_vol``): attached Physical Volume ID (>= 1).
            - Column 1 (``neg_vol``): secondary Physical Volume ID.
                * 0  -> outer boundary triangle.
                * >0 -> shared interface triangle between two volumes.
            - Column 2 (``local_tri_id``): row index of the face in ``faces``.

        4. file_to_ids_dict (dict[str, list[int]]):
            STEP filename -> list of Physical Volume IDs (or per-tet Element
            IDs when ``return_each_tet_as_body=True``).

        5. tet_centroids_data (list[dict]):
            Per-tet centroid data when ``return_each_tet_as_body=True``,
            otherwise ``[]``. Each dict contains:
                - "tet_id" (int)
                - "centroid" (np.ndarray, shape (3,))
                - "volume_tag" (int)
                - "filename" (str)

    Raises:
        ValueError: If a STEP file cannot be read, or no valid 3D volumes
            are detected.

    Example:
        >>> verts, faces, mapping, file_map, _ = glue(
        ...     step_paths=["cone.step", "cube.step"],
        ...     min_size=1.0,
        ...     max_size=5.0,
        ...     return_internal_surfaces=True,
        ... )
        >>> print(f"V={len(verts)}, F={len(faces)}")
        >>> interface_faces = faces[mapping[:, 1] > 0]
    """
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("multi_body_step_mesh")

    if min_size > 0:
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", min_size)
    if max_size > 0 and min_size <= max_size:
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", max_size)

    file_to_original_vols, orig_vol_to_filename = _import_and_index_step_files(
        step_paths
    )

    all_volumes = []
    for vols in file_to_original_vols.values():
        all_volumes.extend(vols)

    # --- fragment: 切开所有互相接触的实体 ---
    frag_vol_to_orig = _fragment_all_volumes(all_volumes)
    new_vol_to_filename = {
        t: orig_vol_to_filename[o] for t, o in frag_vol_to_orig.items()
    }

    # --- 记录质心（removeDup 前）---
    frag_vol_centroids = _record_volume_centroids()

    # --- 合并几何上完全重合的 volume / surface ---
    gmsh.model.occ.removeAllDuplicates()
    gmsh.model.occ.synchronize()

    # --- 用质心重建 tag → filename（tag 已被重排）---
    new_vol_to_filename = _rebuild_volume_to_filename(
        new_vol_to_filename, frag_vol_centroids
    )

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
        # ---- 每个 tet 作为一个 body ----
        elem_types_3d, elem_tags_3d, elem_node_tags_3d = (
            gmsh.model.mesh.getElements(3)
        )

        face_to_vols = {}
        for etype, tags, node_tags_list in zip(
            elem_types_3d, elem_tags_3d, elem_node_tags_3d
        ):
            if etype == 4:  # 4-node tetrahedron
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
        # ---- 按 CAD volume 分组 ----
        for vol_geom_tag, phys_id in vol_tag_to_physical_id.items():
            fname = vol_tag_to_filename.get(vol_geom_tag)
            if fname and fname in file_to_ids_dict:
                file_to_ids_dict[fname].append(phys_id)

        # 建立 surface tag -> 归属的 (vol_phys_id, is_reversed) 列表。
        # 一个 surface 若被 >=2 个 volume 引用，即为 shared interface。
        surf_ownership = {}
        for vol in geom_volumes:
            vol_geom_tag = vol[1]
            vol_phys_id = vol_tag_to_physical_id[vol_geom_tag]
            for b in gmsh.model.getBoundary([vol], combined=False, oriented=True):
                if b[0] == 2:
                    signed_surf_tag = int(b[1])
                    surf_tag = abs(signed_surf_tag)
                    is_reversed = signed_surf_tag < 0
                    surf_ownership.setdefault(surf_tag, []).append(
                        (vol_phys_id, is_reversed)
                    )

        raw_faces = []
        mapping_table = []
        local_tri_id = 0

        for surf_tag, owners in surf_ownership.items():
            # return_internal_surfaces=False: 跳过共享接口面，
            # 只保留外表面，保证整体 watertight。
            if not return_internal_surfaces and len(owners) > 1:
                continue

            elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(
                2, surf_tag
            )
            for i, etype in enumerate(elem_types):
                if etype == 2:  # 3-node triangle
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

    # 重新编号 node tags -> 0-based 连续索引
    unique_node_tags = np.unique(raw_faces)
    vertices = coord_array[unique_node_tags]

    tag_to_idx = {tag: idx for idx, tag in enumerate(unique_node_tags)}
    faces = np.vectorize(tag_to_idx.get)(raw_faces)

    gmsh.finalize()

    return vertices, faces, mapping_table, file_to_ids_dict, tet_centroids_data


# ---------------------------------------------------------------------------
# 导出
# ---------------------------------------------------------------------------

def vf2stl(
    vertices: np.ndarray,
    faces: np.ndarray,
    output_stl_path: str,
    binary: bool = True,
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
    """Convert a Gmsh .msh file to an I-deas .unv file for FreeCAD FEM
    workbench."""
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


def msh2stl(msh_path: str, stl_path: str, include_internal: bool = False) -> str:
    """Extract triangle faces from a 3D tetrahedral .msh file and write ASCII STL.

    Args:
        msh_path: Input .msh file path (must contain 3D tetrahedra).
        stl_path: Output .stl file path.
        include_internal:
            - False (default): export only the outer boundary skin
              (faces shared by exactly 1 tetrahedron). The result is
              watertight.
            - True: export ALL faces, including internal contact
              interfaces between volumes (faces shared by 2 tetrahedra
              coming from different physical volumes). Each internal
              face is emitted once. The result is a conformal
              multi-domain mesh, not watertight at the interfaces.

    Returns:
        Absolute path to the written STL file.
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

        # ---- 收集所有 tet 面（带方向）----
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
                face_count.setdefault(sorted_f, []).append(f)

        # ---- 选择要输出的面 ----
        triangles = []
        n_boundary = 0
        n_internal = 0
        n_nonmanifold = 0
        for sorted_f, f_list in face_count.items():
            n = len(f_list)
            if n == 1:
                triangles.append(f_list[0])
                n_boundary += 1
            elif n == 2:
                if include_internal:
                    triangles.append(f_list[0])  # 内部面只输出一次
                    n_internal += 1
            else:
                # 非流形边（3+ tets 共享一面）——正常网格不会出现
                n_nonmanifold += 1
                if include_internal:
                    triangles.append(f_list[0])

        if n_nonmanifold:
            print(f"Warning: {n_nonmanifold} non-manifold tet faces detected")

        # ---- 写 STL ----
        output_dir = os.path.dirname(os.path.abspath(stl_path))
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        with open(stl_path, "w") as f:
            f.write("solid GmshModel\n")
            for tri in triangles:
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
