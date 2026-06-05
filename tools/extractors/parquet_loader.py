"""
parquet_loader.py
-----------------
从 Habitat parquet 文件加载轨迹数据
"""

import pandas as pd
from typing import List, Dict, Any
from pathlib import Path


class ParquetLoader:
    """Parquet 文件加载器"""

    def __init__(self, parquet_path: str):
        """
        Args:
            parquet_path: parquet 文件路径或文件对象
        """
        self.parquet_path = parquet_path
        self.df = None

    def load(self) -> pd.DataFrame:
        """加载 parquet 文件"""
        if isinstance(self.parquet_path, (str, Path)):
            self.df = pd.read_parquet(self.parquet_path)
        else:
            # 文件对象（从 tar 提取）
            self.df = pd.read_parquet(self.parquet_path)
        return self.df

    def get_frame_count(self) -> int:
        """获取帧数"""
        if self.df is None:
            self.load()
        return len(self.df)

    def get_frame_data(self, frame_index: int) -> Dict[str, Any]:
        """
        获取单帧数据

        Args:
            frame_index: 帧索引

        Returns:
            包含 frame_index, timestamp, pose, action 等的字典
        """
        if self.df is None:
            self.load()

        row = self.df.iloc[frame_index]

        return {
            "frame_index": int(row["frame_index"]),
            "timestamp": float(row["timestamp"]),
            "pose": row["pose.125cm_0deg"],  # 4x4 transformation matrix
            "action": int(row["action"]),
            "goal": row.get("goal.125cm_0deg"),
            "relative_goal_frame_id": int(row.get("relative_goal_frame_id.125cm_0deg", -1))
        }

    def iter_frames(self):
        """迭代所有帧"""
        if self.df is None:
            self.load()

        for _, row in self.df.iterrows():
            yield {
                "frame_index": int(row["frame_index"]),
                "timestamp": float(row["timestamp"]),
                "pose": row["pose.125cm_0deg"],
                "action": int(row["action"]),
                "goal": row.get("goal.125cm_0deg"),
                "relative_goal_frame_id": int(row.get("relative_goal_frame_id.125cm_0deg", -1))
            }
