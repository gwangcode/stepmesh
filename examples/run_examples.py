from pathlib import Path
from stepmesh import step2msh, glue, vf2stl, msh2unv, msh2stl

if __name__ == "__main__":
    # 获取当前脚本所在目录 (examples/ 目录)
    examples_dir = Path(__file__).parent.resolve()

    # 自动定位同目录下的示例 STEP 文件
    step_paths = [
        str(examples_dir / "cone.step"),
        str(examples_dir / "cube.step"),
        str(examples_dir / "cylinder.step")
    ]

    output_msh = str(examples_dir / "output_n_bodies.msh")
    output_stl = str(examples_dir / "test_output.stl")
    output_unv = str(examples_dir / "output_n_bodies.unv")
    output_msh_stl = str(examples_dir / "output_n_bodies.stl")

    print("=== 1. Testing step2msh conversion ===")
    step2msh(step_paths, output_msh)

    print("\n=== 2. Testing glue extraction ===")
    vertices, faces, mapping_table, file_to_ids_dict, tet_centroids_data = glue(
        step_paths, return_each_tet_as_body=True
    )

    print(f"Vertices shape: {vertices.shape}")
    print(f"Faces shape: {faces.shape}")
    print(f"Mapping table shape: {mapping_table.shape}")
    print(f"File to IDs Mapping: {file_to_ids_dict}")
    print(f"Total tetrahedra centroids extracted: {len(tet_centroids_data)}")

    print("\n=== 3. Exporting formatted files ===")
    vf2stl(vertices, faces, output_stl, binary=True)
    msh2unv(output_msh, output_unv)
    msh2stl(output_msh, output_msh_stl)
    print("Example execution completed successfully!")