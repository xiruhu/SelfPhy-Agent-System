"""
batch_extract_and_process.py
-----------------------------
批量提取并处理 Habitat 数据的自动化流水线

功能：
1. 从 tar.gz 提取 episodes → trajectory.json
2. 自动运行 habitat_metadata_builder.py 生成 metadata.json
3. 支持多场景、多 episode 批处理
4. 生成处理报告

运行示例：
  # 处理单个场景
  python batch_extract_and_process.py \
    --tar-dir ../data/VL-LN-Bench/traj_data/mp3d_split2 \
    --output-dir ../data/VL-LN-Bench/processed \
    --scenes D7N2EKCX4Sj \
    --max-episodes 3

  # 处理多个场景
  python batch_extract_and_process.py \
    --tar-dir ../data/VL-LN-Bench/traj_data/mp3d_split2 \
    --output-dir ../data/VL-LN-Bench/processed \
    --scenes D7N2EKCX4Sj q9vSo1VnCiC \
    --max-episodes 5
"""

import sys
import argparse
import subprocess
from pathlib import Path
from typing import List, Dict, Any
import json
from datetime import datetime


class BatchProcessor:
    """批量处理器"""

    def __init__(self, tar_dir: Path, output_dir: Path):
        """
        Args:
            tar_dir: tar.gz 文件所在目录
            output_dir: 输出根目录
        """
        self.tar_dir = tar_dir
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 脚本路径
        self.extract_script = Path(__file__).parent / "extract_all_episodes.py"
        self.metadata_builder = Path(__file__).parent.parent / "core" / "habitat_metadata_builder.py"

    def process_scene(
        self,
        scene_name: str,
        max_episodes: int = None
    ) -> Dict[str, Any]:
        """
        处理单个场景

        Args:
            scene_name: 场景名称 (如 D7N2EKCX4Sj)
            max_episodes: 最多处理的 episode 数量

        Returns:
            处理结果统计
        """
        print(f"\n{'=' * 60}")
        print(f"[场景] {scene_name}")
        print(f"{'=' * 60}\n")

        tar_file = self.tar_dir / f"{scene_name}.tar.gz"

        if not tar_file.exists():
            print(f"[ERROR] tar 文件不存在：{tar_file}")
            return {"success": False, "error": "tar_not_found"}

        scene_output_dir = self.output_dir / scene_name
        scene_output_dir.mkdir(parents=True, exist_ok=True)

        # Step 1: 提取 episodes
        print("[步骤 1/2] 提取 episodes...")

        extract_cmd = [
            sys.executable,
            str(self.extract_script),
            "--tar", str(tar_file),
            "--output", str(scene_output_dir)
        ]

        if max_episodes:
            extract_cmd.extend(["--max-episodes", str(max_episodes)])

        try:
            result = subprocess.run(
                extract_cmd,
                check=True,
                capture_output=True,
                text=True
            )
            print(result.stdout)

        except subprocess.CalledProcessError as e:
            print(f"[ERROR] 提取失败：{e}")
            print(e.stderr)
            return {"success": False, "error": "extraction_failed"}

        # Step 2: 为每个 episode 生成 metadata.json
        print("\n[步骤 2/2] 生成 metadata.json...")

        episode_dirs = sorted(scene_output_dir.glob("episode_*"))
        metadata_count = 0
        metadata_errors = 0

        for episode_dir in episode_dirs:
            trajectory_file = episode_dir / "trajectory.json"

            if not trajectory_file.exists():
                continue

            metadata_file = episode_dir / "metadata.json"

            # 跳过已存在的 metadata
            if metadata_file.exists():
                print(f"  [SKIP] {episode_dir.name}/metadata.json")
                metadata_count += 1
                continue

            # 运行 metadata_builder
            metadata_cmd = [
                sys.executable,
                str(self.metadata_builder),
                str(trajectory_file),
                "-o", str(metadata_file)
            ]

            try:
                subprocess.run(
                    metadata_cmd,
                    check=True,
                    capture_output=True,
                    text=True
                )
                print(f"  [DONE] {episode_dir.name}/metadata.json")
                metadata_count += 1

            except subprocess.CalledProcessError as e:
                print(f"  [ERROR] {episode_dir.name}: {e}")
                metadata_errors += 1

        return {
            "success": True,
            "scene_name": scene_name,
            "episode_count": len(episode_dirs),
            "metadata_count": metadata_count,
            "metadata_errors": metadata_errors
        }

    def process_scenes(
        self,
        scene_names: List[str],
        max_episodes: int = None
    ) -> Dict[str, Any]:
        """
        批量处理多个场景

        Args:
            scene_names: 场景名称列表
            max_episodes: 每个场景最多处理的 episode 数量

        Returns:
            总体处理报告
        """
        report = {
            "start_time": datetime.now().isoformat(),
            "scenes": [],
            "summary": {
                "total_scenes": len(scene_names),
                "success_scenes": 0,
                "failed_scenes": 0,
                "total_episodes": 0,
                "total_metadatas": 0
            }
        }

        for scene_name in scene_names:
            result = self.process_scene(scene_name, max_episodes)
            report["scenes"].append(result)

            if result["success"]:
                report["summary"]["success_scenes"] += 1
                report["summary"]["total_episodes"] += result.get("episode_count", 0)
                report["summary"]["total_metadatas"] += result.get("metadata_count", 0)
            else:
                report["summary"]["failed_scenes"] += 1

        report["end_time"] = datetime.now().isoformat()

        return report


def main():
    parser = argparse.ArgumentParser(
        description="批量提取并处理 Habitat 数据"
    )

    parser.add_argument(
        "--tar-dir",
        required=True,
        help="tar.gz 文件所在目录"
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        help="输出根目录"
    )

    parser.add_argument(
        "--scenes",
        nargs="+",
        required=True,
        help="要处理的场景名称列表"
    )

    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="每个场景最多处理的 episode 数量"
    )

    args = parser.parse_args()

    # 创建处理器
    processor = BatchProcessor(
        Path(args.tar_dir),
        Path(args.output_dir)
    )

    # 批量处理
    report = processor.process_scenes(
        args.scenes,
        args.max_episodes
    )

    # 保存报告
    report_file = Path(args.output_dir) / "batch_processing_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # 打印总结
    print("\n" + "=" * 60)
    print("[总体报告]")
    print(f"  - 处理场景：{report['summary']['total_scenes']}")
    print(f"  - 成功场景：{report['summary']['success_scenes']}")
    print(f"  - 失败场景：{report['summary']['failed_scenes']}")
    print(f"  - 总 episodes：{report['summary']['total_episodes']}")
    print(f"  - 总 metadatas：{report['summary']['total_metadatas']}")
    print(f"  - 报告文件：{report_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()
