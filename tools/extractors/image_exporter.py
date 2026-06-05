"""
image_exporter.py
-----------------
从 tar 中提取 RGB 和 Depth 图像
"""

import tarfile
from pathlib import Path
from typing import Optional, Dict


class ImageExporter:
    """图像提取器"""

    def __init__(self, tar_handle: tarfile.TarFile, scene_name: str):
        """
        Args:
            tar_handle: 打开的 tar 文件句柄
            scene_name: 场景名称 (如 D7N2EKCX4Sj)
        """
        self.tar = tar_handle
        self.scene_name = scene_name

    def build_rgb_path(self, episode_id: int, frame_id: int) -> str:
        """构建 RGB 图像在 tar 中的路径"""
        return (
            f"{self.scene_name}/videos/chunk-000/"
            f"observation.images.rgb.125cm_0deg/"
            f"episode_{episode_id:06d}_{frame_id}.jpg"
        )

    def build_depth_path(self, episode_id: int, frame_id: int) -> str:
        """构建 Depth 图像在 tar 中的路径"""
        return (
            f"{self.scene_name}/videos/chunk-000/"
            f"observation.images.depth.125cm_0deg/"
            f"episode_{episode_id:06d}_{frame_id}.png"
        )

    def extract_file(self, tar_path: str, output_path: Path) -> bool:
        """
        从 tar 中提取单个文件

        Args:
            tar_path: tar 中的文件路径
            output_path: 输出路径

        Returns:
            是否成功
        """
        try:
            member = self.tar.getmember(tar_path)
            file_obj = self.tar.extractfile(member)

            if file_obj is None:
                return False

            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, "wb") as f:
                f.write(file_obj.read())

            return True

        except KeyError:
            # 文件不存在
            return False
        except Exception as e:
            print(f"[ERROR] Failed to extract {tar_path}: {e}")
            return False

    def extract_frame_images(
        self,
        episode_id: int,
        frame_id: int,
        output_dir: Path
    ) -> Dict[str, Optional[str]]:
        """
        提取单帧的 RGB 和 Depth 图像

        Args:
            episode_id: episode ID
            frame_id: 帧 ID
            output_dir: 输出目录

        Returns:
            {"rgb": "rgb/000042.jpg", "depth": "depth/000042.png"} 或 None
        """
        rgb_dir = output_dir / "rgb"
        depth_dir = output_dir / "depth"

        rgb_filename = f"{frame_id:06d}.jpg"
        depth_filename = f"{frame_id:06d}.png"

        rgb_output = rgb_dir / rgb_filename
        depth_output = depth_dir / depth_filename

        # 提取 RGB
        rgb_tar_path = self.build_rgb_path(episode_id, frame_id)
        rgb_ok = self.extract_file(rgb_tar_path, rgb_output)

        # 提取 Depth
        depth_tar_path = self.build_depth_path(episode_id, frame_id)
        depth_ok = self.extract_file(depth_tar_path, depth_output)

        return {
            "rgb": f"rgb/{rgb_filename}" if rgb_ok else None,
            "depth": f"depth/{depth_filename}" if depth_ok else None
        }
