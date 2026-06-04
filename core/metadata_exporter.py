"""
metadata_exporter.py
--------------------
数据链桥接模块：将 cv2_processor 输出的 TrajectorySegment JSON
转换为 claude_examiner 所需的扁平 metadata 列表。

数据流：
  outputs/trajectories/*.json  →  [本模块]  →  outputs/metadata/metadata.json

每条 metadata 记录字段：
  frame_name     : 帧文件名（用于溯源）
  episode_id     : 所属 episode
  image_path     : 关键帧图片绝对/相对路径
  yaw            : 相对上一关键帧的 yaw 变化量（度），正=右转，负=左转
  pitch          : 相对上一关键帧的 pitch 变化量（度）
  roll           : 相对上一关键帧的 roll 变化量（度）
  motion_score   : 帧间运动强度（位移 * 10 + |yaw|，量纲统一）
  displacement_m : 相对上一关键帧的位移（米）
  abs_yaw        : 当前帧的绝对 yaw 角（度）
  action_label   : 动作标签（forward/turn_left/turn_right 等）
  frame_id       : 帧序号
  timestamp      : 时间戳（秒）
"""

import json
import sys
from pathlib import Path


TRAJECTORIES_DIR = Path("outputs/trajectories")
METADATA_OUTPUT  = Path("outputs/metadata/metadata.json")
METADATA_OUTPUT.parent.mkdir(parents=True, exist_ok=True)


def euler_diff(curr: list, prev: list) -> tuple[float, float, float]:
    """计算两帧欧拉角的差值 [roll, pitch, yaw]（度）。"""
    if not curr or not prev:
        return 0.0, 0.0, 0.0
    roll  = curr[0] - prev[0]
    pitch = curr[1] - prev[1]
    yaw   = curr[2] - prev[2]
    # 归一化到 [-180, 180]
    yaw   = (yaw + 180) % 360 - 180
    pitch = (pitch + 180) % 360 - 180
    return roll, pitch, yaw


def compute_displacement(pos_curr: list, pos_prev: list) -> float:
    """计算两帧之间的欧氏位移（米）。"""
    if not pos_curr or not pos_prev:
        return 0.0
    dx = pos_curr[0] - pos_prev[0]
    dy = pos_curr[1] - pos_prev[1]
    dz = pos_curr[2] - pos_prev[2]
    return (dx**2 + dy**2 + dz**2) ** 0.5


def motion_score(displacement_m: float, abs_yaw_diff: float) -> float:
    """
    综合运动强度分数。
    公式：displacement_m * 10 + |yaw_diff|
    与原 gpt4o_examiner 的 motion_score 阈值（15）兼容。
    """
    return displacement_m * 10.0 + abs(abs_yaw_diff)


def export_trajectory(traj_path: Path) -> list[dict]:
    """
    解析单个 TrajectorySegment JSON，返回该 episode 的 metadata 列表。
    第一帧（无前驱帧）的 yaw/motion_score 均为 0，不会触发出题阈值，自然跳过。
    """
    with open(traj_path, "r", encoding="utf-8") as f:
        traj = json.load(f)

    episode_id = traj.get("segment_id", traj_path.stem)
    keyframes  = traj.get("keyframes", [])

    records = []
    prev_euler = None
    prev_pos   = None

    for kf in keyframes:
        pose       = kf.get("pose", {})
        euler      = pose.get("euler_angles")   # [roll, pitch, yaw] 度
        position   = pose.get("position")       # [x, y, z] 米
        frame_id   = kf.get("frame_id", 0)
        image_path = kf.get("image_path", "")
        frame_name = Path(image_path).name if image_path else f"frame_{frame_id:04d}.jpg"
        timestamp  = pose.get("timestamp", 0.0)
        action     = pose.get("action_label", "unknown")

        # 计算帧间变化量
        if prev_euler is not None and prev_pos is not None:
            roll_d, pitch_d, yaw_d = euler_diff(euler, prev_euler)
            disp = compute_displacement(position, prev_pos)
        else:
            roll_d, pitch_d, yaw_d, disp = 0.0, 0.0, 0.0, 0.0

        score = motion_score(disp, yaw_d)

        records.append({
            "episode_id":     episode_id,
            "frame_name":     frame_name,
            "image_path":     image_path,
            "frame_id":       frame_id,
            "timestamp":      timestamp,
            "yaw":            round(yaw_d, 2),
            "pitch":          round(pitch_d, 2),
            "roll":           round(roll_d, 2),
            "abs_yaw":        round(euler[2], 2) if euler else 0.0,
            "displacement_m": round(disp, 3),
            "motion_score":   round(score, 2),
            "action_label":   action,
        })

        prev_euler = euler
        prev_pos   = position

    return records


def main():
    traj_files = sorted(TRAJECTORIES_DIR.glob("*.json"))

    if not traj_files:
        print(f"[ERROR] 在 {TRAJECTORIES_DIR} 下未找到任何轨迹文件。")
        print("请先运行 cv2_processor.py（或 create_sample_data.py + cv2_processor.py）生成轨迹数据。")
        sys.exit(1)

    all_records: list[dict] = []

    for traj_path in traj_files:
        print(f"  [解析] {traj_path.name}")
        records = export_trajectory(traj_path)
        all_records.extend(records)
        print(f"         → {len(records)} 帧记录")

    with open(METADATA_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(all_records, f, ensure_ascii=False, indent=2)

    print(f"\n[完成] 共导出 {len(all_records)} 条记录 → {METADATA_OUTPUT}")

    # 统计会触发出题的帧数（与 claude_examiner 阈值一致）
    triggered = [r for r in all_records if abs(r["yaw"]) > 20 and r["motion_score"] > 15]
    print(f"[统计] 满足出题阈值（|yaw|>20° & motion_score>15）的帧：{len(triggered)} 条")


if __name__ == "__main__":
    main()
