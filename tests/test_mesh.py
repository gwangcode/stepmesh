import os
from pathlib import Path
import pytest
import numpy as np
from stepmesh import step2msh, glue, vf2stl, msh2unv, msh2stl


@pytest.fixture(scope="module")
def sample_step_paths():
    """获取测试用的 STEP 文件路径，如果不存在则跳过测试"""
    base_dir = Path(__file__).parent.parent / "examples"
    paths = [
        str(base_dir / "cone.step"),
        str(base_dir / "cube.step"),
        str(base_dir / "cylinder.step"),
    ]

    # 检查测试文件是否存在
    for p in paths:
        if not os.path.exists(p):
            pytest.skip(f"Test STEP file not found: {p}")

    return paths


@pytest.fixture
def temp_output_dir(tmp_path):
    """使用 pytest 内置的临时目录保存测试输出文件"""
    return tmp_path


def test_glue_basic(sample_step_paths):
    """测试 glue 函数的基本抽取逻辑与数组维度"""
    vertices, faces, mapping_table, file_to_ids_dict = glue(
        sample_step_paths,
        min_size=1.0,
        max_size=3.0,
        return_internal_surfaces=False
    )

    assert isinstance(vertices, np.ndarray)
    assert isinstance(faces, np.ndarray)
    assert isinstance(mapping_table, np.ndarray)
    assert isinstance(file_to_ids_dict, dict)

    assert vertices.ndim == 2 and vertices.shape[1] == 3
    assert faces.ndim == 2 and faces.shape[1] == 3
    assert mapping_table.ndim == 2 and mapping_table.shape[1] == 3
    assert len(vertices) > 0
    assert len(faces) > 0


def test_glue_each_tet_as_body(sample_step_paths):
    """测试开启 return_each_tet_as_body=True 时的输出"""
    result = glue(
        sample_step_paths,
        min_size=1.0,
        max_size=3.0,
        return_each_tet_as_body=True
    )

    assert len(result) == 5
    vertices, faces, mapping_table, file_to_ids_dict, tet_centroids_data = result

    assert isinstance(tet_centroids_data, list)
    assert len(tet_centroids_data) > 0

    # 验证形心数据格式
    first_tet = tet_centroids_data[0]
    assert "tet_id" in first_tet
    assert "centroid" in first_tet
    assert len(first_tet["centroid"]) == 3


def test_step2msh(sample_step_paths, temp_output_dir):
    """测试 STEP 转 MSH 文件写入"""
    output_msh = str(temp_output_dir / "test_out.msh")
    file_to_ids = step2msh(sample_step_paths, output_msh, min_size=1.0, max_size=3.0)

    assert os.path.exists(output_msh)
    assert os.path.getsize(output_msh) > 0
    assert isinstance(file_to_ids, dict)


def test_format_conversions(sample_step_paths, temp_output_dir):
    """测试 STL 与 UNV 格式转换工具"""
    output_msh = str(temp_output_dir / "test_out.msh")
    output_unv = str(temp_output_dir / "test_out.unv")
    output_stl = str(temp_output_dir / "test_out.stl")
    custom_stl = str(temp_output_dir / "custom_vf.stl")

    # 1. 先生成 msh 文件
    step2msh(sample_step_paths, output_msh, min_size=1.0, max_size=3.0)

    # 2. 测试 msh2unv 与 msh2stl
    unv_res = msh2unv(output_msh, output_unv)
    stl_res = msh2stl(output_msh, output_stl)

    assert os.path.exists(unv_res) and os.path.getsize(unv_res) > 0
    assert os.path.exists(stl_res) and os.path.getsize(stl_res) > 0

    # 3. 测试 vf2stl 直接导出数组
    vertices, faces, _, _ = glue(sample_step_paths, min_size=1.0, max_size=3.0)
    vf2stl(vertices, faces, custom_stl, binary=True)
    assert os.path.exists(custom_stl) and os.path.getsize(custom_stl) > 0


def test_invalid_file_handling(temp_output_dir):
    """测试输入不存在的 STEP 文件时的异常拦截"""
    non_existent_file = [str(temp_output_dir / "not_exist.step")]

    with pytest.raises(ValueError, match="No valid 3D volumes detected"):
        step2msh(non_existent_file, str(temp_output_dir / "should_fail.msh"))