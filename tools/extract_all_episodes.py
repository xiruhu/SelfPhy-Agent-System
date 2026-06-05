"""
extract_all_episodes.py (V2)
-----------------------------
批量提取 Habitat episodes，生成 metadata_builder 兼容的 trajectory.json

功能：
1. 自动扫描 tar 中所有 parquet 文件
2. 提取 RGB + Depth 图像
3. 导出标准化的 trajectory.json（兼容 metadata_builder）
4. 支持断点续传
5. 支持 --max-episodes 限制处理数量

运行示例：
  # 提取单个 episode
  python extract_all_episodes.py \
    --tar ../data/VL-LN-Bench/traj_data/mp3d_split2/D7N2EKCX4Sj.tar.gz \
    --output ../data/VL-LN-Bench/extracted/D7N2EKCX4Sj \
    --max-episodes 1

  # 提取所有 episodes
  python extract_all_episodes.py \
    --tar ../data/VL-LN-Bench/traj_data/mp3d_split2/D7N2EKCX4Sj.tar.gz \
    --output ../data/VL-LN-Bench/extracted/D7N2EKCX4Sj
"""

import os
import re
import sys
import tarfile
import argparse
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from extractors.episode_exporter import EpisodeExporter


def get_episode_id_from_parquet(parquet_path: str) -> int:
    """
    从 parquet 路径提取 episode ID
    例如：episode_000058.parquet -> 58
    """
    match = re.search(r"episode_(\d+)\.parquet", parquet_path)
    if match is None:
        return None
    return int(match.group(1))


def main():
    parser = argparse.ArgumentParser(
        description="批量提取 Habitat episodes 为标准化的 trajectory.json"
    )

    parser.add_argument(
        "--tar",
        required=True,
        help="Habitat 场景的 tar.gz 文件路径"
    )

    parser.add_argument(
        "--output",
        required=True,
        help="输出根目录"
    )

    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="最多处理的 episode 数量（用于测试）"
    )

    parser.add_argument(
        "--no-images",
        action="store_true",
        help="不提取图像，只生成 trajectory.json"
    )

    args = parser.parse_args()

    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] 打开 tar 文件：{args.tar}")
    tar = tarfile.open(args.tar, "r:gz")

    # 获取场景名称（从第一个成员推断）
    members = tar.getnames()
    scene_name = members[0].split("/")[0]
    print(f"[INFO] 场景名称：{scene_name}")

    # 查找所有 parquet 文件
    parquet_members = [
        m for m in members
        if m.endswith(".parquet") and "/data/" in m
    ]
    parquet_members.sort()

    print(f"[INFO] 找到 {len(parquet_members)} 个 episodes")

    # 限制处理数量
    if args.max_episodes:
        parquet_members = parquet_members[:args.max_episodes]
        print(f"[INFO] 只处理前 {args.max_episodes} 个 episodes")

    # 创建导出器
    exporter = EpisodeExporter(tar, scene_name)

    # 处理每个 episode
    success_count = 0
    skip_count = 0
    error_count = 0

    for parquet_member in parquet_members:
        episode_id = get_episode_id_from_parquet(parquet_member)

        if episode_id is None:
            print(f"[WARN] 无法解析 episode ID：{parquet_member}")
            continue

        episode_dir = output_root / f"episode_{episode_id:06d}"
        trajectory_file = episode_dir / "trajectory.json"

        # 断点续传：如果已存在则跳过
        if trajectory_file.exists():
            print(f"[SKIP] Episode {episode_id:06d}（已存在）")
            skip_count += 1
            continue

        print(f"\n[PROCESS] Episode {episode_id:06d}")

        try:
            # 提取 parquet 文件
            parquet_file = tar.extractfile(parquet_member)

            if parquet_file is None:
                print(f"[ERROR] 无法打开 parquet：{parquet_member}")
                error_count += 1
                continue

            # 导出 episode
            result = exporter.export_episode(
                episode_id,
                parquet_file,
                episode_dir,
                extract_images=not args.no_images
            )

            print(f"[DONE] Episode {episode_id:06d}")
            print(f"  - 帧数：{result['frame_count']}")
            print(f"  - 时长：{result['duration']:.2f}s")
            print(f"  - trajectory.json：{result['trajectory_path']}")

            success_count += 1

        except Exception as e:
            print(f"[ERROR] 处理 Episode {episode_id:06d} 时出错：{e}")
            import traceback
            traceback.print_exc()
            error_count += 1

    # 关闭 tar
    tar.close()

    # 总结
    print("\n" + "=" * 60)
    print("[总结]")
    print(f"  - 成功：{success_count} 个")
    print(f"  - 跳过：{skip_count} 个")
    print(f"  - 失败：{error_count} 个")
    print(f"  - 总计：{success_count + skip_count + error_count} 个")
    print("=" * 60)


if __name__ == "__main__":
    main()