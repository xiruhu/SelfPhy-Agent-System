"""
读取 habitat_collector.py 的数据采集结果
"""

import numpy as np
from scipy.spatial.transform import Rotation
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import json
import cv2
from dataclasses import dataclass


@dataclass
class HabitatFrame:
    """单帧 Habitat 数据"""
    frame_id: int
    timestamp: float
    rgb: np.ndarray
    depth: Optional[np.ndarray]
    position: np.ndarray  # [x, y, z]
    rotation: np.ndarray  # quaternion [qw, qx, qy, qz]


class HabitatDataLoader:
    """Habitat 数据加载器"""

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)

    def load_episode(self, episode_id: str) -> Dict[str, Any]:
        """
        加载一个 episode 的完整数据

        Args:
            episode_id: episode 标识符

        Returns:
            包含 frames, metadata 的字典
        """
        episode_dir = self.data_dir / episode_id

        if not episode_dir.exists():
            raise FileNotFoundError(f"Episode directory not found: {episode_dir}")

        # 读取位姿数据
        poses_path = episode_dir / "poses.json"
        with open(poses_path, 'r') as f:
            poses_data = json.load(f)

        # 读取元数据
        metadata_path = episode_dir / "metadata.json"
        metadata = {}
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

        # 加载视频帧
        frames = self._load_frames(episode_dir, poses_data)

        return {
            "episode_id": episode_id,
            "frames": frames,
            "metadata": metadata
        }

    def _load_frames(self, episode_dir: Path, poses_data: List[Dict]) -> List[HabitatFrame]:
        """加载所有帧"""
        frames = []
        frames_dir = episode_dir / "frames"

        if not frames_dir.exists():
            # 如果没有预提取的帧，从视频中提取
            video_path = episode_dir / "video.mp4"
            if video_path.exists():
                frames = self._extract_frames_from_video(video_path, poses_data)
            else:
                raise FileNotFoundError(f"No frames or video found in {episode_dir}")
        else:
            # 从预提取的帧中加载
            for i, pose in enumerate(poses_data):
                frame_path = frames_dir / f"frame_{i:04d}.jpg"
                if frame_path.exists():
                    rgb = cv2.imread(str(frame_path))
                    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

                    frame = HabitatFrame(
                        frame_id=i,
                        timestamp=pose.get('timestamp', i * 0.033),  # 假设 30fps
                        rgb=rgb,
                        depth=None,
                        position=np.array(pose['position']),
                        rotation=np.array(pose['rotation'])
                    )
                    frames.append(frame)

        return frames

    def _extract_frames_from_video(self, video_path: Path, poses_data: List[Dict]) -> List[HabitatFrame]:
        """从视频中提取帧"""
        cap = cv2.VideoCapture(str(video_path))
        frames = []

        frame_idx = 0
        while cap.isOpened() and frame_idx < len(poses_data):
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pose = poses_data[frame_idx]

            habitat_frame = HabitatFrame(
                frame_id=frame_idx,
                timestamp=pose.get('timestamp', frame_idx * 0.033),
                rgb=rgb,
                depth=None,
                position=np.array(pose['position']),
                rotation=np.array(pose['rotation'])
            )
            frames.append(habitat_frame)
            frame_idx += 1

        cap.release()
        return frames

    def quaternion_to_euler(self, quat: np.ndarray) -> np.ndarray:
        """
        四元数转欧拉角

        Args:
            quat: 四元数 [qw, qx, qy, qz] 或 [qx, qy, qz, qw]

        Returns:
            欧拉角 [roll, pitch, yaw] (度数)
        """
        # Habitat 使用 [qw, qx, qy, qz] 格式
        # scipy 使用 [qx, qy, qz, qw] 格式
        if len(quat) == 4:
            # 转换为 scipy 格式
            quat_scipy = [quat[1], quat[2], quat[3], quat[0]]
            r = Rotation.from_quat(quat_scipy)
            euler = r.as_euler('xyz', degrees=True)
            return euler
        else:
            raise ValueError(f"Invalid quaternion format: {quat}")

    def compute_rotation_diff(self, quat1: np.ndarray, quat2: np.ndarray) -> float:
        """
        计算两个四元数之间的旋转角度

        Args:
            quat1, quat2: 四元数 [qw, qx, qy, qz]

        Returns:
            旋转角度（度数）
        """
        # 转换为 scipy 格式
        q1_scipy = [quat1[1], quat1[2], quat1[3], quat1[0]]
        q2_scipy = [quat2[1], quat2[2], quat2[3], quat2[0]]

        r1 = Rotation.from_quat(q1_scipy)
        r2 = Rotation.from_quat(q2_scipy)

        # 计算相对旋转
        r_diff = r2 * r1.inv()
        angle = r_diff.magnitude() * 180 / np.pi

        return angle

    def compute_translation_distance(self, pos1: np.ndarray, pos2: np.ndarray) -> float:
        """计算两个位置之间的欧氏距离"""
        return np.linalg.norm(pos2 - pos1)

    def extract_keyframe_indices(
        self,
        frames: List[HabitatFrame],
        rotation_threshold: float = 30.0,
        translation_threshold: float = 1.0
    ) -> List[int]:
        """
        提取关键帧索引

        Args:
            frames: 帧列表
            rotation_threshold: 旋转角度阈值（度）
            translation_threshold: 平移距离阈值（米）

        Returns:
            关键帧索引列表
        """
        if not frames:
            return []

        keyframe_indices = [0]  # 第一帧总是关键帧

        for i in range(1, len(frames)):
            prev_frame = frames[keyframe_indices[-1]]  # 与上一个关键帧比较
            curr_frame = frames[i]

            # 计算旋转差异
            rotation_diff = self.compute_rotation_diff(
                prev_frame.rotation,
                curr_frame.rotation
            )

            # 计算平移距离
            translation_diff = self.compute_translation_distance(
                prev_frame.position,
                curr_frame.position
            )

            # 判断是否为关键帧
            if rotation_diff > rotation_threshold or translation_diff > translation_threshold:
                keyframe_indices.append(i)

        return keyframe_indices

    def generate_spatial_narrative(
        self,
        frames: List[HabitatFrame],
        keyframe_indices: List[int]
    ) -> str:
        """
        生成自然语言轨迹描述

        Args:
            frames: 所有帧
            keyframe_indices: 关键帧索引

        Returns:
            自然语言描述
        """
        if len(keyframe_indices) < 2:
            return "Agent remained stationary."

        narrative_parts = []

        for i in range(1, len(keyframe_indices)):
            prev_idx = keyframe_indices[i-1]
            curr_idx = keyframe_indices[i]

            prev_frame = frames[prev_idx]
            curr_frame = frames[curr_idx]

            # 计算位移
            displacement = curr_frame.position - prev_frame.position
            distance = np.linalg.norm(displacement)

            # 计算旋转
            rotation_angle = self.compute_rotation_diff(
                prev_frame.rotation,
                curr_frame.rotation
            )

            # 判断主要运动方向
            if distance > 0.5:
                # 有明显平移
                direction = self._get_movement_direction(displacement)
                narrative_parts.append(
                    f"moved {direction} for {distance:.1f}m"
                )

            if abs(rotation_angle) > 15:
                # 有明显旋转
                turn_direction = "right" if rotation_angle > 0 else "left"
                narrative_parts.append(
                    f"turned {turn_direction} by {abs(rotation_angle):.0f}°"
                )

        if narrative_parts:
            return "Agent " + ", then ".join(narrative_parts) + "."
        else:
            return "Agent made minor adjustments."

    def _get_movement_direction(self, displacement: np.ndarray) -> str:
        """根据位移向量判断运动方向"""
        x, y, z = displacement

        # 主要看 x 和 z 方向（假设 y 是高度）
        if abs(z) > abs(x):
            return "forward" if z > 0 else "backward"
        else:
            return "right" if x > 0 else "left"

    def save_episode_data(
        self,
        episode_id: str,
        frames: List[HabitatFrame],
        output_dir: Path
    ) -> None:
        """
        保存 episode 数据到磁盘

        Args:
            episode_id: episode 标识符
            frames: 帧列表
            output_dir: 输出目录
        """
        episode_dir = output_dir / episode_id
        episode_dir.mkdir(parents=True, exist_ok=True)

        # 保存帧图像
        frames_dir = episode_dir / "frames"
        frames_dir.mkdir(exist_ok=True)

        for frame in frames:
            frame_path = frames_dir / f"frame_{frame.frame_id:04d}.jpg"
            rgb_bgr = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(frame_path), rgb_bgr)

        # 保存位姿数据
        poses_data = []
        for frame in frames:
            poses_data.append({
                "frame_id": frame.frame_id,
                "timestamp": frame.timestamp,
                "position": frame.position.tolist(),
                "rotation": frame.rotation.tolist()
            })

        poses_path = episode_dir / "poses.json"
        with open(poses_path, 'w') as f:
            json.dump(poses_data, f, indent=2)
