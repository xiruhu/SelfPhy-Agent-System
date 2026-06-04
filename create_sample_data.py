"""
创建示例 Habitat 数据
用于测试系统（当没有真实 Habitat 数据时）
"""

import json
import numpy as np
import cv2
from pathlib import Path
from scipy.spatial.transform import Rotation


def create_sample_episode(episode_id: str = "episode_001", num_frames: int = 30):
    """
    创建示例 episode 数据

    Args:
        episode_id: episode 标识符
        num_frames: 帧数
    """
    output_dir = Path("data/raw/habitat") / episode_id
    output_dir.mkdir(parents=True, exist_ok=True)

    frames_dir = output_dir / "frames"
    frames_dir.mkdir(exist_ok=True)

    print(f"创建示例数据: {episode_id}")

    # 生成模拟轨迹
    poses = []

    # 起点
    position = np.array([0.0, 0.0, 0.0])
    yaw = 0.0  # 朝向角度（度）

    for i in range(num_frames):
        # 模拟运动
        if i % 10 == 5:
            # 每 10 帧转一次弯
            yaw += np.random.choice([-90, 90])

        # 前进
        forward_distance = np.random.uniform(0.1, 0.3)
        position[0] += forward_distance * np.sin(np.radians(yaw))
        position[2] += forward_distance * np.cos(np.radians(yaw))

        # 转换为四元数
        r = Rotation.from_euler('y', yaw, degrees=True)
        quat = r.as_quat()  # [qx, qy, qz, qw]
        quat_habitat = [quat[3], quat[0], quat[1], quat[2]]  # [qw, qx, qy, qz]

        # 保存位姿
        poses.append({
            "frame_id": i,
            "timestamp": i * 0.033,  # 30 FPS
            "position": position.tolist(),
            "rotation": quat_habitat
        })

        # 生成示例图像（彩色噪声）
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        # 添加一些简单的"场景"元素
        # 绘制地平线
        cv2.line(img, (0, 240), (640, 240), (100, 150, 200), 2)

        # 绘制一些"物体"（矩形）
        for _ in range(3):
            x = np.random.randint(50, 590)
            y = np.random.randint(100, 400)
            w = np.random.randint(30, 80)
            h = np.random.randint(30, 80)
            color = tuple(np.random.randint(50, 255, 3).tolist())
            cv2.rectangle(img, (x, y), (x+w, y+h), color, -1)

        # 添加帧号
        cv2.putText(
            img,
            f"Frame {i}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )

        # 保存图像
        frame_path = frames_dir / f"frame_{i:04d}.jpg"
        cv2.imwrite(str(frame_path), img)

    # 保存位姿数据
    poses_path = output_dir / "poses.json"
    with open(poses_path, 'w') as f:
        json.dump(poses, f, indent=2)

    # 保存元数据
    metadata = {
        "episode_id": episode_id,
        "num_frames": num_frames,
        "fps": 30,
        "resolution": [640, 480],
        "scene": "sample_scene",
        "description": "Synthetic sample data for testing"
    }

    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"✓ 创建完成: {num_frames} 帧")
    print(f"  位置: {output_dir}")


def main():
    """创建多个示例 episode"""
    import argparse

    parser = argparse.ArgumentParser(description="创建示例 Habitat 数据")
    parser.add_argument("--num-episodes", type=int, default=3, help="创建的 episode 数量")
    parser.add_argument("--num-frames", type=int, default=30, help="每个 episode 的帧数")
    args = parser.parse_args()

    print("=" * 80)
    print("  创建示例 Habitat 数据")
    print("=" * 80 + "\n")

    # 创建指定数量的示例 episode
    for i in range(1, args.num_episodes + 1):
        episode_id = f"episode_{i:03d}"
        num_frames = args.num_frames if args.num_frames > 0 else np.random.randint(20, 40)
        create_sample_episode(episode_id, num_frames)
        print()

    print("=" * 80)
    print(f"  示例数据创建完成 ({args.num_episodes} episodes)")
    print("=" * 80)
    print("\n现在可以运行: python run_pipeline.py")


if __name__ == "__main__":
    main()
