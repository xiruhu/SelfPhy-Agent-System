"""
episode_exporter.py
-------------------
Episode 导出器：从 parquet 提取轨迹，导出为 metadata_builder 兼容的格式
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import sys

sys.path.append(str(Path(__file__).parent))

from parquet_loader import ParquetLoader
from pose_utils import pose_matrix_to_position_quaternion
from image_exporter import ImageExporter


class EpisodeExporter:
    """Episode 数据导出器"""

    def __init__(self, tar_handle, scene_name: str):
        """
        Args:
            tar_handle: tarfile.TarFile 对象
            scene_name: 场景名称 (如 D7N2EKCX4Sj)
        """
        self.tar = tar_handle
        self.scene_name = scene_name
        self.image_exporter = ImageExporter(tar_handle, scene_name)

    def export_episode(
        self,
        episode_id: int,
        parquet_file_or_path,
        output_dir: Path,
        extract_images: bool = True
    ) -> Dict[str, Any]:
        """
        导出单个 episode

        Args:
            episode_id: episode ID
            parquet_file_or_path: parquet 文件路径或文件对象
            output_dir: 输出目录
            extract_images: 是否提取图像

        Returns:
            {"trajectory_path": ..., "metadata_path": ..., "frame_count": ...}
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # 加载 parquet
        loader = ParquetLoader(parquet_file_or_path)
        loader.load()

        # 构建 trajectory（兼容 metadata_builder 格式）
        trajectory = []

        for frame_data in loader.iter_frames():
            frame_id = frame_data["frame_index"]
            timestamp = frame_data["timestamp"]
            pose_matrix = frame_data["pose"]

            # 转换 pose matrix → position + quaternion
            position, quaternion = pose_matrix_to_position_quaternion(pose_matrix)

            frame_info = {
                "frame_id": frame_id,
                "timestamp": timestamp,
                "position": position,
                "rotation": quaternion,  # [w, x, y, z]
                "action": frame_data["action"]
            }

            # 提取图像
            if extract_images:
                image_paths = self.image_exporter.extract_frame_images(
                    episode_id,
                    frame_id,
                    output_dir
                )
                frame_info["rgb"] = image_paths["rgb"]
                frame_info["depth"] = image_paths["depth"]

            trajectory.append(frame_info)

        # 保存 trajectory.json（兼容 metadata_builder）
        trajectory_json = {
            "episode_id": episode_id,
            "scene_id": self.scene_name,
            "trajectory": trajectory  # 关键：使用 "trajectory" 键
        }

        trajectory_path = output_dir / "trajectory.json"
        with open(trajectory_path, "w", encoding="utf-8") as f:
            json.dump(trajectory_json, f, indent=2, ensure_ascii=False)

        # 保存简单的 metadata
        metadata = {
            "episode_id": episode_id,
            "scene_id": self.scene_name,
            "num_frames": len(trajectory),
            "duration": trajectory[-1]["timestamp"] - trajectory[0]["timestamp"]
        }

        metadata_path = output_dir / "episode_info.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        return {
            "trajectory_path": str(trajectory_path),
            "metadata_path": str(metadata_path),
            "frame_count": len(trajectory),
            "duration": metadata["duration"]
        }
