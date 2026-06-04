"""
数据感知层 (Data Perception Module)
负责从 Habitat 数据中提取关键帧、构建轨迹、检测物体
"""

import cv2
import os
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import sys

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from core.habitat_loader import HabitatDataLoader, HabitatFrame


@dataclass
class Pose6DoF:
    """6自由度位姿"""
    frame_id: int
    timestamp: float
    position: Tuple[float, float, float]
    orientation: Tuple[float, float, float, float]  # quaternion [qw, qx, qy, qz]
    euler_angles: Optional[Tuple[float, float, float]] = None  # [roll, pitch, yaw] 度数
    action_label: Optional[str] = None


@dataclass
class DetectedObject:
    """检测到的物体"""
    object_id: str
    class_name: str
    bbox: Tuple[int, int, int, int]  # [x1, y1, x2, y2]
    confidence: float
    attributes: Optional[Dict[str, Any]] = None


@dataclass
class KeyFrame:
    """关键帧数据包"""
    frame_id: int
    image_path: str
    pose: Pose6DoF
    detected_objects: List[DetectedObject]
    scene_description: Optional[str] = None


@dataclass
class TrajectorySegment:
    """轨迹片段"""
    segment_id: str
    video_source: str
    start_time: float
    end_time: float
    keyframes: List[KeyFrame]
    spatial_narrative: str
    metadata: Optional[Dict[str, Any]] = None


class DataPerceptionModule:
    """数据感知层"""

    def __init__(self, output_dir: str = "outputs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.frames_dir = self.output_dir / "frames"
        self.trajectories_dir = self.output_dir / "trajectories"

        self.frames_dir.mkdir(exist_ok=True)
        self.trajectories_dir.mkdir(exist_ok=True)

        self.habitat_loader = HabitatDataLoader("data/raw/habitat")

    def process_episode(
        self,
        episode_id: str,
        rotation_threshold: float = 30.0,
        translation_threshold: float = 1.0
    ) -> TrajectorySegment:
        """
        处理一个 episode，提取关键帧并构建轨迹

        Args:
            episode_id: episode 标识符
            rotation_threshold: 旋转角度阈值（度）
            translation_threshold: 平移距离阈值（米）

        Returns:
            轨迹片段
        """
        print(f"[Processing] Episode: {episode_id}")

        # 加载 episode 数据
        episode_data = self.habitat_loader.load_episode(episode_id)
        frames = episode_data['frames']

        # 提取关键帧索引
        keyframe_indices = self.habitat_loader.extract_keyframe_indices(
            frames,
            rotation_threshold=rotation_threshold,
            translation_threshold=translation_threshold
        )

        print(f"[Extracted] {len(keyframe_indices)} keyframes from {len(frames)} frames")

        # 构建关键帧数据
        keyframes = []
        episode_frames_dir = self.frames_dir / episode_id
        episode_frames_dir.mkdir(exist_ok=True)

        for idx in keyframe_indices:
            frame = frames[idx]

            # 保存关键帧图像
            frame_filename = f"frame_{frame.frame_id:04d}.jpg"
            frame_path = episode_frames_dir / frame_filename

            rgb_bgr = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(frame_path), rgb_bgr)

            # 转换四元数为欧拉角
            euler = self.habitat_loader.quaternion_to_euler(frame.rotation)

            # 构建 Pose6DoF
            pose = Pose6DoF(
                frame_id=frame.frame_id,
                timestamp=frame.timestamp,
                position=tuple(frame.position.tolist()),
                orientation=tuple(frame.rotation.tolist()),
                euler_angles=tuple(euler.tolist()),
                action_label=self._infer_action_label(frames, idx, keyframe_indices)
            )

            # 检测物体（简化版，实际应使用 YOLO/GroundingDINO）
            detected_objects = self._detect_objects_simple(frame.rgb)

            # 构建 KeyFrame
            keyframe = KeyFrame(
                frame_id=frame.frame_id,
                image_path=str(frame_path),
                pose=pose,
                detected_objects=detected_objects,
                scene_description=None
            )

            keyframes.append(keyframe)

        # 生成自然语言轨迹描述
        spatial_narrative = self.habitat_loader.generate_spatial_narrative(
            frames,
            keyframe_indices
        )

        # 构建轨迹片段
        trajectory = TrajectorySegment(
            segment_id=episode_id,
            video_source=f"habitat/{episode_id}",
            start_time=frames[0].timestamp,
            end_time=frames[-1].timestamp,
            keyframes=keyframes,
            spatial_narrative=spatial_narrative,
            metadata=episode_data.get('metadata', {})
        )

        # 保存轨迹
        self._save_trajectory(trajectory)

        print(f"[Completed] Trajectory saved: {episode_id}")

        return trajectory

    def _infer_action_label(
        self,
        frames: List[HabitatFrame],
        current_idx: int,
        keyframe_indices: List[int]
    ) -> str:
        """推断动作标签"""
        if current_idx == 0:
            return "start"

        # 找到上一个关键帧
        prev_keyframe_idx = None
        for i, kf_idx in enumerate(keyframe_indices):
            if kf_idx == current_idx and i > 0:
                prev_keyframe_idx = keyframe_indices[i-1]
                break

        if prev_keyframe_idx is None:
            return "unknown"

        prev_frame = frames[prev_keyframe_idx]
        curr_frame = frames[current_idx]

        # 计算旋转和平移
        rotation_diff = self.habitat_loader.compute_rotation_diff(
            prev_frame.rotation,
            curr_frame.rotation
        )
        translation_diff = self.habitat_loader.compute_translation_distance(
            prev_frame.position,
            curr_frame.position
        )

        # 判断主要动作
        if translation_diff > 0.5:
            displacement = curr_frame.position - prev_frame.position
            if abs(displacement[2]) > abs(displacement[0]):
                return "forward" if displacement[2] > 0 else "backward"
            else:
                return "move_right" if displacement[0] > 0 else "move_left"
        elif abs(rotation_diff) > 15:
            return "turn_right" if rotation_diff > 0 else "turn_left"
        else:
            return "minor_adjustment"

    def _detect_objects_simple(self, rgb: np.ndarray) -> List[DetectedObject]:
        """
        简化的物体检测（占位符）
        实际应使用 YOLO、GroundingDINO 或 Habitat 的语义标注
        """
        # TODO: 集成真实的物体检测模型
        # 这里返回空列表作为占位
        return []

    def _save_trajectory(self, trajectory: TrajectorySegment) -> None:
        """保存轨迹到 JSON 文件"""
        output_path = self.trajectories_dir / f"{trajectory.segment_id}.json"

        # 转换为可序列化的字典
        trajectory_dict = {
            "segment_id": trajectory.segment_id,
            "video_source": trajectory.video_source,
            "start_time": trajectory.start_time,
            "end_time": trajectory.end_time,
            "spatial_narrative": trajectory.spatial_narrative,
            "keyframes": [
                {
                    "frame_id": kf.frame_id,
                    "image_path": kf.image_path,
                    "pose": asdict(kf.pose),
                    "detected_objects": [asdict(obj) for obj in kf.detected_objects],
                    "scene_description": kf.scene_description
                }
                for kf in trajectory.keyframes
            ],
            "metadata": trajectory.metadata
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(trajectory_dict, f, indent=2, ensure_ascii=False)

    def visualize_trajectory(
        self,
        trajectory: TrajectorySegment,
        output_path: Optional[str] = None
    ) -> str:
        """
        可视化轨迹（俯视图）

        Args:
            trajectory: 轨迹片段
            output_path: 输出路径（可选）

        Returns:
            图片保存路径
        """
        import matplotlib.pyplot as plt

        if output_path is None:
            output_path = self.output_dir / "visualizations" / f"{trajectory.segment_id}_trajectory.png"
        else:
            output_path = Path(output_path)

        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 提取位置数据
        positions = [kf.pose.position for kf in trajectory.keyframes]
        x_coords = [pos[0] for pos in positions]
        z_coords = [pos[2] for pos in positions]

        # 绘制轨迹
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.plot(x_coords, z_coords, 'b-', linewidth=2, label='Trajectory')
        ax.scatter(x_coords, z_coords, c='red', s=100, zorder=5, label='Keyframes')

        # 标注起点和终点
        ax.scatter(x_coords[0], z_coords[0], c='green', s=200, marker='*', zorder=6, label='Start')
        ax.scatter(x_coords[-1], z_coords[-1], c='orange', s=200, marker='X', zorder=6, label='End')

        # 添加箭头指示方向
        for i in range(len(x_coords) - 1):
            dx = x_coords[i+1] - x_coords[i]
            dz = z_coords[i+1] - z_coords[i]
            ax.arrow(x_coords[i], z_coords[i], dx*0.8, dz*0.8,
                    head_width=0.1, head_length=0.1, fc='blue', ec='blue', alpha=0.5)

        ax.set_xlabel('X (meters)', fontsize=12)
        ax.set_ylabel('Z (meters)', fontsize=12)
        ax.set_title(f'Trajectory: {trajectory.segment_id}', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axis('equal')

        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()

        print(f"[Saved] Trajectory visualization: {output_path}")

        return str(output_path)


def main():
    """主函数：处理示例 episode"""
    processor = DataPerceptionModule()

    # 示例：处理一个 episode
    episode_id = "episode_001"

    try:
        trajectory = processor.process_episode(
            episode_id=episode_id,
            rotation_threshold=30.0,
            translation_threshold=1.0
        )

        # 可视化轨迹
        processor.visualize_trajectory(trajectory)

        print(f"\n[Success] Processed episode: {episode_id}")
        print(f"Keyframes: {len(trajectory.keyframes)}")
        print(f"Narrative: {trajectory.spatial_narrative}")

    except Exception as e:
        print(f"[Error] Failed to process episode: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()