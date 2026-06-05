"""
habitat_metadata_builder.py
----------------------------
从 Habitat trajectory.json（RGB+Depth+Pose）提取 yaw/displacement 生成 metadata.json

数据流：
  extract_all_episodes.py → trajectory.json (精确的Habitat数据)
        ↓
  habitat_metadata_builder.py (本模块)
        ↓
  metadata.json (yaw/displacement/motion_score)
        ↓
  claude_examiner.py
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
import sys


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


def compute_displacement(pos1: List[float], pos2: List[float]) -> float:
    """
    计算两个位置之间的欧式距离

    Args:
        pos1, pos2: [x, y, z] 坐标

    Returns:
        距离（米）
    """
    return np.linalg.norm(np.array(pos2) - np.array(pos1))


def compute_motion_score(
    delta_yaw: float,
    displacement: float,
    time_delta: float
) -> float:
    """
    计算运动强度分数

    Args:
        delta_yaw: 旋转角度变化（度）
        displacement: 位移距离（米）
        time_delta: 时间间隔（秒）

    Returns:
        运动分数（0-100）
    """
    # 归一化旋转分量（假设180度为满分）
    rotation_score = min(abs(delta_yaw) / 180.0, 1.0) * 50

    # 归一化位移分量（假设2米为满分）
    translation_score = min(displacement / 2.0, 1.0) * 50

    return rotation_score + translation_score


def extract_keyframes(
    trajectory: List[Dict],
    yaw_threshold: float = 20.0,
    displacement_threshold: float = 0.5,
    motion_score_threshold: float = 15.0
) -> List[Dict]:
    """
    从完整轨迹中提取关键帧

    Args:
        trajectory: 完整轨迹数据
        yaw_threshold: yaw变化阈值（度）
        displacement_threshold: 位移阈值（米）
        motion_score_threshold: 运动分数阈值

    Returns:
        关键帧列表
    """
    if not trajectory:
        return []

    keyframes = []

    # 第一帧总是关键帧
    first_frame = trajectory[0].copy()

    # 计算第一帧的欧拉角
    rot = first_frame["rotation"]
    roll, pitch, yaw = quaternion_to_euler(rot[0], rot[1], rot[2], rot[3])
    first_frame["euler_angles"] = [roll, pitch, yaw]
    first_frame["yaw"] = yaw
    first_frame["displacement_from_prev"] = 0.0
    first_frame["motion_score"] = 0.0

    keyframes.append(first_frame)

    prev_keyframe = first_frame

    # 遍历后续帧
    for i in range(1, len(trajectory)):
        current_frame = trajectory[i].copy()

        # 计算当前帧的欧拉角
        rot = current_frame["rotation"]
        roll, pitch, yaw = quaternion_to_euler(rot[0], rot[1], rot[2], rot[3])
        current_frame["euler_angles"] = [roll, pitch, yaw]
        current_frame["yaw"] = yaw

        # 计算与上一个关键帧的差异
        delta_yaw = abs(yaw - prev_keyframe["yaw"])
        displacement = compute_displacement(
            prev_keyframe["position"],
            current_frame["position"]
        )
        time_delta = current_frame["timestamp"] - prev_keyframe["timestamp"]

        # 计算运动分数
        motion_score = compute_motion_score(delta_yaw, displacement, time_delta)

        current_frame["displacement_from_prev"] = displacement
        current_frame["motion_score"] = motion_score
        current_frame["delta_yaw"] = delta_yaw

        # 判断是否为关键帧
        is_keyframe = (
            delta_yaw >= yaw_threshold or
            displacement >= displacement_threshold or
            motion_score >= motion_score_threshold
        )

        if is_keyframe:
            keyframes.append(current_frame)
            prev_keyframe = current_frame

    return keyframes


def build_metadata(
    trajectory_json_path: str,
    output_path: str = None,
    yaw_threshold: float = 20.0,
    displacement_threshold: float = 0.5,
    motion_score_threshold: float = 15.0
) -> Dict:
    """
    从 trajectory.json 构建 metadata.json

    Args:
        trajectory_json_path: trajectory.json 路径
        output_path: 输出路径（可选）
        yaw_threshold: yaw变化阈值
        displacement_threshold: 位移阈值
        motion_score_threshold: 运动分数阈值

    Returns:
        metadata 字典
    """
    # 加载 trajectory.json
    with open(trajectory_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    trajectory = data.get("trajectory", data) if isinstance(data, dict) else data

    # 提取关键帧
    keyframes = extract_keyframes(
        trajectory,
        yaw_threshold=yaw_threshold,
        displacement_threshold=displacement_threshold,
        motion_score_threshold=motion_score_threshold
    )

    # 计算统计信息
    total_displacement = sum(
        compute_displacement(trajectory[i]["position"], trajectory[i+1]["position"])
        for i in range(len(trajectory) - 1)
    )

    total_rotation = 0.0
    for i in range(len(trajectory) - 1):
        rot1 = trajectory[i]["rotation"]
        rot2 = trajectory[i+1]["rotation"]
        _, _, yaw1 = quaternion_to_euler(rot1[0], rot1[1], rot1[2], rot1[3])
        _, _, yaw2 = quaternion_to_euler(rot2[0], rot2[1], rot2[2], rot2[3])
        total_rotation += abs(yaw2 - yaw1)

    # 构建 metadata
    metadata = {
        "episode_id": data.get("episode_id", "unknown"),
        "scene_id": data.get("scene_id", "unknown"),
        "total_frames": len(trajectory),
        "keyframes_count": len(keyframes),
        "total_duration": trajectory[-1]["timestamp"] - trajectory[0]["timestamp"],
        "total_displacement": float(total_displacement),
        "total_rotation": float(total_rotation),
        "keyframes": keyframes,
        "extraction_params": {
            "yaw_threshold": yaw_threshold,
            "displacement_threshold": displacement_threshold,
            "motion_score_threshold": motion_score_threshold
        }
    }

    # 保存到文件
    if output_path:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        print(f"✅ Metadata saved to {output_file}")
        print(f"   Total frames: {len(trajectory)}")
        print(f"   Keyframes: {len(keyframes)}")
        print(f"   Total displacement: {total_displacement:.2f}m")
        print(f"   Total rotation: {total_rotation:.1f}°")

    return metadata


def batch_build_metadata(
    input_dir: str,
    output_dir: str,
    pattern: str = "**/trajectory.json",
    **kwargs
):
    """
    批量处理多个 trajectory.json

    Args:
        input_dir: 输入目录
        output_dir: 输出目录
        pattern: 文件匹配模式
        **kwargs: 传递给 build_metadata 的参数
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    trajectory_files = list(input_path.glob(pattern))

    print(f"Found {len(trajectory_files)} trajectory files")

    for traj_file in trajectory_files:
        # 生成对应的输出路径
        relative_path = traj_file.relative_to(input_path)
        output_file = output_path / relative_path.parent / "metadata.json"

        print(f"\nProcessing: {traj_file}")

        try:
            build_metadata(str(traj_file), str(output_file), **kwargs)
        except Exception as e:
            print(f"❌ Error processing {traj_file}: {e}")
            continue


def main():
    """主函数：命令行接口"""
    import argparse

    parser = argparse.ArgumentParser(description="从 Habitat trajectory.json 构建 metadata.json")
    parser.add_argument("input", help="trajectory.json 文件路径或目录")
    parser.add_argument("-o", "--output", help="输出路径")
    parser.add_argument("--yaw-threshold", type=float, default=20.0, help="yaw变化阈值（度）")
    parser.add_argument("--displacement-threshold", type=float, default=0.5, help="位移阈值（米）")
    parser.add_argument("--motion-score-threshold", type=float, default=15.0, help="运动分数阈值")
    parser.add_argument("--batch", action="store_true", help="批量处理模式")

    args = parser.parse_args()

    if args.batch:
        # 批量处理
        output_dir = args.output or "outputs/metadata"
        batch_build_metadata(
            args.input,
            output_dir,
            yaw_threshold=args.yaw_threshold,
            displacement_threshold=args.displacement_threshold,
            motion_score_threshold=args.motion_score_threshold
        )
    else:
        # 单文件处理
        output_path = args.output or args.input.replace("trajectory.json", "metadata.json")
        build_metadata(
            args.input,
            output_path,
            yaw_threshold=args.yaw_threshold,
            displacement_threshold=args.displacement_threshold,
            motion_score_threshold=args.motion_score_threshold
        )


if __name__ == "__main__":
    main()
