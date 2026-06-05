"""
pose_utils.py
-------------
位姿转换工具：4x4矩阵 → position + rotation/quaternion
"""

import numpy as np
from typing import List, Tuple


def pose_matrix_to_position_rotation(pose_matrix) -> Tuple[List[float], List[List[float]]]:
    """
    从 4x4 pose matrix 提取 position 和 3x3 rotation matrix

    Args:
        pose_matrix: 4x4 transformation matrix (可能是嵌套列表、numpy数组列表等)

    Returns:
        (position [x, y, z], rotation [[r00, r01, r02], [r10, r11, r12], [r20, r21, r22]])
    """
    # 处理多种可能的输入格式
    if isinstance(pose_matrix, np.ndarray):
        # 如果已经是 numpy 数组
        if pose_matrix.dtype == object:
            # 如果是 object 类型（嵌套的 numpy 数组），需要手动堆叠
            pose = np.vstack([np.array(row).flatten() for row in pose_matrix]).reshape(4, 4)
        else:
            pose = pose_matrix
    elif isinstance(pose_matrix, list):
        # 如果是列表，检查元素类型
        if len(pose_matrix) > 0 and isinstance(pose_matrix[0], np.ndarray):
            # 列表中包含 numpy 数组，需要堆叠
            pose = np.vstack([row.flatten() for row in pose_matrix]).reshape(4, 4)
        else:
            # 普通嵌套列表
            pose = np.array(pose_matrix)
    else:
        pose = np.array(pose_matrix)

    # 确保是 2D 数组
    if pose.ndim == 1:
        if len(pose) == 16:
            pose = pose.reshape(4, 4)
        else:
            raise ValueError(f"Unexpected pose shape: {pose.shape}, cannot reshape to 4x4")

    if pose.shape != (4, 4):
        raise ValueError(f"Expected 4x4 matrix, got shape {pose.shape}")

    # 提取平移向量 (最后一列的前三行)
    position = pose[:3, 3].tolist()

    # 提取旋转矩阵 (左上 3x3)
    rotation = pose[:3, :3].tolist()

    return position, rotation


def rotation_matrix_to_quaternion(rotation: List[List[float]]) -> List[float]:
    """
    3x3 rotation matrix → quaternion [w, x, y, z]

    Args:
        rotation: 3x3 rotation matrix

    Returns:
        quaternion [w, x, y, z]
    """
    R = np.array(rotation)

    trace = np.trace(R)

    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    return [float(w), float(x), float(y), float(z)]


def pose_matrix_to_position_quaternion(pose_matrix: List[List[float]]) -> Tuple[List[float], List[float]]:
    """
    从 4x4 pose matrix 提取 position 和 quaternion

    Args:
        pose_matrix: 4x4 transformation matrix

    Returns:
        (position [x, y, z], quaternion [w, x, y, z])
    """
    position, rotation = pose_matrix_to_position_rotation(pose_matrix)
    quaternion = rotation_matrix_to_quaternion(rotation)
    return position, quaternion


def quaternion_to_euler(qw: float, qx: float, qy: float, qz: float) -> Tuple[float, float, float]:
    """
    四元数转欧拉角 (roll, pitch, yaw)

    Args:
        qw, qx, qy, qz: 四元数分量

    Returns:
        (roll, pitch, yaw) in degrees
    """
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (qw * qx + qy * qz)
    cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2 * (qw * qy - qz * qx)
    if abs(sinp) >= 1:
        pitch = np.copysign(np.pi / 2, sinp)
    else:
        pitch = np.arcsin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.degrees(roll), np.degrees(pitch), np.degrees(yaw)
