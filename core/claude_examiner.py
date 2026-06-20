"""
claude_examiner.py
---------------------
SelfPhy-Agent-System 的考试生成器 V2（Supervisor Agent 模块）

核心变化：
1. 长官模型（Claude）只出题，不描述场景
2. 基于 Habitat 精确数据（RGB + Depth + Pose Matrix）
3. 输出纯问题 + 多模态证据，强制被测模型理解第一人称视觉

数据流：
  metadata.json (来自 habitat_metadata_builder.py)
        ↓
  [本模块] Claude Sonnet 4.6 生成纯问题
        ↓
  exam_v2.json (QuestionV2 格式)
"""

import json
import os
import sys
import base64
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

import anthropic
from dotenv import load_dotenv

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from schema.question import (
    QuestionV2,
    ExamPaperV2,
    MultimodalEvidence,
    Pose6DoF
)

# ─────────────────────────────────────────────
# 环境与路径配置
# ─────────────────────────────────────────────

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
ANTHROPIC_BASE_URL = os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com")

if not ANTHROPIC_API_KEY:
    sys.exit("[ERROR] 未找到 ANTHROPIC_API_KEY，请检查 .env 文件。")

# 长官模型：claude-sonnet-4-6
SUPERVISOR_MODEL = "claude-sonnet-4-6"

# ─────────────────────────────────────────────
# 初始化 Anthropic 客户端
# ─────────────────────────────────────────────

client = anthropic.Anthropic(
    api_key=ANTHROPIC_API_KEY,
    base_url=ANTHROPIC_BASE_URL,
)

# ─────────────────────────────────────────────
# System Prompt V2
# ─────────────────────────────────────────────

SYSTEM_PROMPT_V2 = """你是 SelfPhy-Agent-System 的考题生成器（Examiner Agent V2）。

【核心原则 - 非常重要】
你的任务是基于第一人称视觉轨迹生成考题，但**绝对不能在问题中描述场景**。

❌ 错误示例（不要这样做）：
"我原本面对一扇门，随后向右转90度，现在看到一张桌子，请问门现在位于什么方向？"
→ 问题：被测模型只需要文本推理，不需要理解视频

✅ 正确示例（必须这样做）：
"最开始位于你左侧的物体现在在哪个方向？"
→ 被测模型必须：
  1. 看 frame_0 识别"左侧有什么"
  2. 看 frame_15-45 理解"我在转身/移动"
  3. 建立空间记忆映射，推断物体现在的相对方向

【输入信息】
你会收到：
1. metadata.json：关键帧信息（frame_id, yaw, displacement, position, rotation）
2. 关键帧图像（RGB，可能包含 Depth）
3. 完整的位姿轨迹（精确的 Habitat 数据）

【输出要求】
生成 JSON 格式的考题列表，每个考题包含：

```json
{
  "question_id": "Q001_egocentric_memory",
  "question_text": "最开始位于你左侧的沙发现在在哪个方向？",
  "ground_truth_answer": "右后方",
  "capability": "egocentric_memory",
  "evidence_frame_ids": [0, 15, 30, 45],
  "trajectory_window": [
    {"frame_id": 0, "position": [0, 0, 0], "rotation": [1, 0, 0, 0], "timestamp": 0.0},
    {"frame_id": 45, "position": [2.3, 0, 1.1], "rotation": [0.7, 0, 0.7, 0], "timestamp": 1.5}
  ],
  "reasoning_trace": "frame_0: 沙发在左侧(-90°)。agent从frame_0到frame_45右转90°，沙发相对新朝向为-180°（右后方）。",
  "difficulty": "medium",
  "rotation_degree": 90.0,
  "displacement_meters": 2.5
}
```

【能力分类与出题模板】

1. **egocentric_memory**（第一人称记忆）
   - 问题形式："最开始X在哪里？" / "现在X在哪个方向？"
   - 考察：模型是否记住起始状态，并理解自我运动后的相对变化

2. **spatial_transformation**（空间变换理解）
   - 问题形式："转身后，原来左边的物体现在在哪？"
   - 考察：模型是否理解自我旋转导致的坐标系变换

3. **occlusion_reasoning**（遮挡推理）
   - 问题形式："被墙遮挡前，你看到的桌子上有几个物体？"
   - 考察：模型是否能在遮挡前建立记忆，遮挡后仍能推理

4. **trajectory_backtracking**（轨迹回溯）
   - 问题形式："如果你沿当前方向后退2米，你会回到最初的门附近吗？"
   - 考察：模型是否理解整个轨迹，能做反向推理

5. **distance_estimation**（距离估算）
   - 问题形式："你现在离最初看到的沙发有多远？"
   - 考察：模型是否能从位移累积估算距离

【难度分级】
- easy: 单次旋转/平移，< 45°或 < 1米，2-3帧
- medium: 组合运动，45-90°或 1-2米，3-4帧
- hard: 复杂轨迹，> 90°或 > 2米，4+帧

【关键约束】
1. 每个问题必须强制模型看多帧（至少2帧）
2. 问题不能包含任何场景描述（物体、方向、动作）
3. 答案必须基于精确的位姿计算，写出推理过程
4. evidence_frame_ids 必须是实际需要的帧
5. trajectory_window 只包含关键转折点的位姿

【输出格式】
严格输出 JSON，不添加任何 markdown 代码块或解释。
"""


# ─────────────────────────────────────────────
# 辅助函数
# ─────────────────────────────────────────────

def encode_image_to_base64(image_path: str) -> str:
    """将图像编码为 base64"""
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


def load_keyframe_images(
    keyframes: List[Dict],
    base_dir: Path,
    max_images: int = 6
) -> List[Dict[str, Any]]:
    """
    加载关键帧图像

    Args:
        keyframes: 关键帧元数据列表
        base_dir: 基础目录（包含 RGB/Depth）
        max_images: 最多加载多少张图像

    Returns:
        图像数据列表 [{"frame_id": ..., "rgb": ..., "depth": ...}, ...]
    """
    images = []

    # 均匀采样关键帧
    if len(keyframes) > max_images:
        indices = [int(i * len(keyframes) / max_images) for i in range(max_images)]
    else:
        indices = list(range(len(keyframes)))

    for idx in indices:
        kf = keyframes[idx]
        frame_id = kf["frame_id"]

        # 构建图像路径
        rgb_path = base_dir / "rgb" / f"episode_{frame_id:06d}.jpg"
        depth_path = base_dir / "depth" / f"episode_{frame_id:06d}.png"

        image_data = {
            "frame_id": frame_id,
            "rgb_path": str(rgb_path) if rgb_path.exists() else None,
            "depth_path": str(depth_path) if depth_path.exists() else None
        }

        images.append(image_data)

    return images


def _angle_diff(a: float, b: float) -> float:
    """有符号最短角度差 a - b，归一化到 (-180, 180]，处理 ±180° wraparound。"""
    return (a - b + 180.0) % 360.0 - 180.0


def compute_cumulative_trajectory(keyframes: List[Dict]) -> List[Dict]:
    """
    从关键帧列表计算累计旋转和累计位移，用于精确出题。

    cum_yaw_delta 通过逐帧累加 angle_diff 得到，正确处理跨越 ±180° 的轨迹。
    delta_yaw 字段优先使用 metadata 中已存的有符号值（由修复后的 builder 写入）；
    若缺失则回退到 angle_diff 计算。

    Returns:
        每帧附带 cum_yaw_delta（从起始帧累计转向量）和 cum_displacement 的列表
    """
    result = []
    cum_yaw = 0.0
    cum_disp = 0.0
    prev_yaw = keyframes[0].get("yaw", 0.0) if keyframes else 0.0

    for i, kf in enumerate(keyframes):
        cur_yaw = kf.get("yaw", 0.0)

        if i == 0:
            step_yaw = 0.0
        else:
            # 优先使用 metadata 中已有的有符号 delta_yaw（builder 已修复）
            # 若为旧数据（绝对值），回退到 angle_diff 重新计算
            stored = kf.get("delta_yaw", None)
            if stored is not None and abs(stored) <= 180.0:
                step_yaw = stored
            else:
                step_yaw = _angle_diff(cur_yaw, prev_yaw)

        cum_yaw += step_yaw
        cum_disp += kf.get("displacement_from_prev", 0.0)

        result.append({
            "frame_id": kf["frame_id"],
            "timestamp": round(kf["timestamp"], 3),
            "yaw": round(cur_yaw, 2),
            "cum_yaw_delta": round(cum_yaw, 2),
            "cum_displacement": round(cum_disp, 3),
            "delta_yaw": round(step_yaw, 2),
            "pos": [round(v, 3) for v in kf["position"]],
        })
        prev_yaw = cur_yaw

    return result


def build_examiner_prompt(
    metadata: Dict,
    num_questions: int = 5
) -> str:
    """
    构建给长官模型的 Prompt。

    核心改进：把完整的关键帧位姿序列（含累计旋转/位移）全部传入，
    要求 Claude 基于精确数值计算答案，不允许估算。
    """
    keyframes = metadata["keyframes"]

    # 计算完整的累计轨迹数据
    traj = compute_cumulative_trajectory(keyframes)

    # 起始和终止状态摘要
    first = traj[0]
    last = traj[-1]
    total_cum_yaw = last["cum_yaw_delta"]
    total_cum_disp = last["cum_displacement"]

    prompt = f"""# 任务：为第一人称空间推理生成考题

## 精确轨迹数据（完整关键帧序列，共 {len(traj)} 帧）

每行格式：frame_id | timestamp(s) | yaw绝对值(°) | 相对起始的累计转向(°) | 累计位移(m) | 与上帧yaw差(°) | 绝对坐标[x,y,z]

```
{chr(10).join(
    f"frame {r['frame_id']:>3} | t={r['timestamp']:>6.3f}s | yaw={r['yaw']:>8.2f}° | cum_yaw={r['cum_yaw_delta']:>8.2f}° | cum_disp={r['cum_displacement']:>6.3f}m | Δyaw={r['delta_yaw']:>7.2f}° | pos={r['pos']}"
    for r in traj
)}
```

## 轨迹总览
- 起始 yaw: {first['yaw']}°，终止 yaw: {last['yaw']}°
- 从起始到终止的累计转向: **{total_cum_yaw:.2f}°**（正=顺时针/右转，负=逆时针/左转）
- 累计位移: **{total_cum_disp:.3f} 米**
- 关键帧总数: {len(traj)}

## 如何利用上述数据精确出题

**空间变换计算规则**（坐标系：yaw=0° 时朝向+Z轴，yaw 增大为右转）：
- 若物体初始在 agent 正前方（0°），agent 右转 θ° 后，物体在 agent 坐标系中变为 **-θ°**（即左移 θ°）
- 物体相对 agent 的方向 = 物体初始方向 - agent 的 cum_yaw_delta
- 方向→方位词：0°=正前，90°=正右，180°/-180°=正后，-90°=正左
- 介于两方位之间时用"右前方"/"左后方"等表达，并标注角度

**出题要点**：
1. 选择一个具体的 frame 区间（如 frame_0 → frame_15），读出该区间的 cum_yaw_delta
2. 设定物体初始方向（如"正前方=0°"、"正右方=90°"）
3. 用公式计算出物体在终止帧的相对方向，这就是精确答案
4. reasoning_trace 必须写出完整计算过程（初始方向 - cum_yaw_delta = 最终方向）

## 要求
生成 {num_questions} 个考题，覆盖：
1. egocentric_memory（至少1题）
2. spatial_transformation（至少1题）
3. trajectory_backtracking 或 distance_estimation（至少1题）

**关键约束**：
- 问题文本不得包含任何场景描述（不能说具体物体名称、动作描述）
- 只能用抽象指代："最开始在你正前方的物体"、"你的右侧的某物体"
- evidence_frame_ids 填写该题需要待测模型观察的帧范围（连续区间，如 [0,1,...,15]）
- ground_truth_answer 必须基于上方精确数值计算，不得估算
- reasoning_trace 必须包含完整数值推导（写出用了哪个帧区间、cum_yaw_delta 是多少、最终角度是多少）

直接输出 JSON 数组，禁止 markdown 代码块：
[
  {{
    "question_id": "Q001_egocentric_memory",
    "question_text": "（不含场景描述的纯问题）",
    "ground_truth_answer": "（精确方位词 + 角度，如：正左方，-90°）",
    "capability": "egocentric_memory",
    "evidence_frame_ids": [0, 1, 2, ..., N],
    "trajectory_window": [],
    "reasoning_trace": "初始方向=X°，frame_0到frame_N的cum_yaw_delta=Y°，最终方向=X-Y=Z°，对应方位：W",
    "difficulty": "easy|medium|hard",
    "rotation_degree": Y,
    "displacement_meters": D
  }}
]
"""
    return prompt


def call_claude_examiner(
    metadata: Dict,
    image_data: List[Dict],
    num_questions: int = 5,
    temperature: float = 0.7
) -> List[QuestionV2]:
    """
    调用 Claude API 生成考题

    Args:
        metadata: metadata.json 内容
        image_data: 图像数据列表
        num_questions: 生成题目数量
        temperature: 温度参数

    Returns:
        QuestionV2 对象列表
    """
    # 构建 Prompt
    user_prompt = build_examiner_prompt(metadata, num_questions)

    # 准备消息内容（文本 + 图像）
    message_content = [
        {
            "type": "text",
            "text": user_prompt
        }
    ]

    # 添加图像
    for img in image_data:
        if img["rgb_path"] and Path(img["rgb_path"]).exists():
            rgb_b64 = encode_image_to_base64(img["rgb_path"])
            message_content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": rgb_b64
                }
            })

    print(f"[Examiner] 调用 Claude API，生成 {num_questions} 个考题...")

    # 调用 API
    response = client.messages.create(
        model=SUPERVISOR_MODEL,
        max_tokens=8000,
        temperature=temperature,
        thinking={
            "type": "enabled",
            "budget_tokens": 5000
        },
        system=SYSTEM_PROMPT_V2,
        messages=[
            {
                "role": "user",
                "content": message_content
            }
        ]
    )

    # 解析响应
    response_text = ""
    for block in response.content:
        if block.type == "text":
            response_text += block.text

    # 提取 JSON：找第一个 [ 到最后一个 ] 之间的内容，处理嵌套结构
    try:
        questions_data = json.loads(response_text)
    except json.JSONDecodeError:
        start = response_text.find('[')
        end = response_text.rfind(']')
        if start != -1 and end != -1 and end > start:
            try:
                questions_data = json.loads(response_text[start:end+1])
            except json.JSONDecodeError:
                # 逐字符匹配平衡括号
                depth = 0
                json_end = start
                for i, ch in enumerate(response_text[start:], start):
                    if ch == '[':
                        depth += 1
                    elif ch == ']':
                        depth -= 1
                        if depth == 0:
                            json_end = i
                            break
                try:
                    questions_data = json.loads(response_text[start:json_end+1])
                except Exception as e:
                    raise ValueError(f"无法从 Claude 响应中提取 JSON：{response_text[:500]}") from e
        else:
            raise ValueError(f"无法从 Claude 响应中提取 JSON：{response_text[:500]}")

    # 转换为 QuestionV2 对象
    questions = []
    for q_data in questions_data:
        # 转换 trajectory_window
        trajectory_window = []
        for pose in q_data.get("trajectory_window", []):
            # 容错处理：如果 pose 是简单的 frame_id，从 metadata 中查找完整信息
            if isinstance(pose, int):
                # pose 是 frame_id，从 metadata 中查找
                frame_id = pose
                keyframe = next((kf for kf in metadata.get("keyframes", []) if kf["frame_id"] == frame_id), None)
                if keyframe:
                    trajectory_window.append(Pose6DoF(
                        frame_id=keyframe["frame_id"],
                        position=keyframe["position"],
                        rotation=keyframe["rotation"],
                        euler_angles=keyframe.get("euler_angles"),
                        timestamp=keyframe["timestamp"]
                    ))
            elif isinstance(pose, dict):
                # pose 是完整的字典
                trajectory_window.append(Pose6DoF(
                    frame_id=pose.get("frame_id", 0),
                    position=pose.get("position", [0, 0, 0]),
                    rotation=pose.get("rotation", [1, 0, 0, 0]),
                    euler_angles=pose.get("euler_angles"),
                    timestamp=pose.get("timestamp", 0.0)
                ))

        question = QuestionV2(
            question_id=q_data["question_id"],
            question_text=q_data["question_text"],
            ground_truth_answer=q_data["ground_truth_answer"],
            capability=q_data["capability"],
            evidence_frame_ids=q_data["evidence_frame_ids"],
            trajectory_window=trajectory_window,
            depth_frame_ids=q_data.get("depth_frame_ids"),
            reasoning_trace=q_data["reasoning_trace"],
            difficulty=q_data["difficulty"],
            spatial_transform_type=q_data.get("spatial_transform_type"),
            rotation_degree=q_data.get("rotation_degree"),
            displacement_meters=q_data.get("displacement_meters"),
            expected_error_patterns=q_data.get("expected_error_patterns"),
            metadata=q_data.get("metadata")
        )

        questions.append(question)

    print(f"[Examiner] 成功生成 {len(questions)} 个考题")

    return questions


def generate_exam_v2(
    metadata_path: str,
    data_dir: str,
    output_path: str = None,
    num_questions: int = 5,
    max_images: int = 6
) -> ExamPaperV2:
    """
    生成 V2 版本考卷

    Args:
        metadata_path: metadata.json 路径
        data_dir: 数据目录（包含 rgb/, depth/ 等）
        output_path: 输出路径
        num_questions: 生成题目数量
        max_images: 最多发送多少张图像给 Claude

    Returns:
        ExamPaperV2 对象
    """
    # 加载 metadata
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    episode_id = metadata.get("episode_id", "unknown")
    scene_id = metadata.get("scene_id", "unknown")

    print(f"\n[Examiner V2] 开始为 {episode_id} 生成考题...")
    print(f"  Scene: {scene_id}")
    print(f"  Keyframes: {metadata['keyframes_count']}")
    print(f"  Total displacement: {metadata['total_displacement']:.2f}m")
    print(f"  Total rotation: {metadata['total_rotation']:.1f}°")

    # 加载关键帧图像
    data_path = Path(data_dir)
    image_data = load_keyframe_images(
        metadata["keyframes"],
        data_path,
        max_images=max_images
    )

    print(f"[Examiner V2] 加载了 {len(image_data)} 张关键帧图像")

    # 调用 Claude 生成考题
    questions = call_claude_examiner(
        metadata,
        image_data,
        num_questions=num_questions
    )

    # 构建 MultimodalEvidence
    # 使用【全部关键帧】，不仅限于 evidence_frame_ids
    # 理由：Claude 出题时需要看运动变化显著的关键帧，保证出题质量
    rgb_frames = {}
    depth_frames = {}
    trajectory = []

    for kf in metadata["keyframes"]:
        frame_id = kf["frame_id"]
        # 使用 metadata 中记录的相对路径，拼接绝对路径
        rgb_rel = kf.get("rgb")
        if rgb_rel:
            rgb_path = data_path / rgb_rel
            if rgb_path.exists():
                rgb_frames[str(frame_id)] = str(rgb_path)

        depth_rel = kf.get("depth")
        if depth_rel:
            depth_path = data_path / depth_rel
            if depth_path.exists():
                depth_frames[str(frame_id)] = str(depth_path)

        # 位姿使用真实值（用于后续诊断，不传给待测模型）
        trajectory.append(Pose6DoF(
            frame_id=frame_id,
            position=kf["position"],
            rotation=kf["rotation"],
            euler_angles=kf.get("euler_angles"),
            timestamp=kf["timestamp"]
        ))

    multimodal_evidence = MultimodalEvidence(
        rgb_frames=rgb_frames,
        depth_frames=depth_frames if depth_frames else None,
        trajectory=trajectory,
        scene_id=scene_id,
        episode_id=str(episode_id)  # 转换为字符串
    )

    # 计算统计信息
    difficulty_dist = {}
    capability_dist = {}
    for q in questions:
        difficulty_dist[q.difficulty] = difficulty_dist.get(q.difficulty, 0) + 1
        capability_dist[q.capability] = capability_dist.get(q.capability, 0) + 1

    # 构建 ExamPaperV2
    exam = ExamPaperV2(
        exam_id=f"exam_{episode_id}",
        trajectory_id=str(episode_id),  # 转换为字符串
        scene_id=scene_id,
        episode_id=str(episode_id),  # 转换为字符串
        questions=questions,
        multimodal_evidence=multimodal_evidence,
        created_at=datetime.now(),
        supervisor_model=SUPERVISOR_MODEL,
        total_questions=len(questions),
        difficulty_distribution=difficulty_dist,
        capability_distribution=capability_dist,
        statistics={
            "total_frames": metadata["total_frames"],
            "keyframes_count": metadata["keyframes_count"],
            "total_displacement": metadata["total_displacement"],
            "total_rotation": metadata["total_rotation"],
            "total_duration": metadata["total_duration"]
        }
    )

    # 保存到文件
    if output_path:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # 转换为 dict 以便 JSON 序列化
        exam_dict = exam.model_dump(mode='json')

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(exam_dict, f, indent=2, ensure_ascii=False)

        print(f"\n✅ 考卷已保存: {output_file}")
        print(f"   题目数量: {len(questions)}")
        print(f"   难度分布: {difficulty_dist}")
        print(f"   能力分布: {capability_dist}")

    return exam


def main():
    """主函数：命令行接口"""
    import argparse

    parser = argparse.ArgumentParser(description="Claude Examiner V2 - 生成纯问题考题")
    parser.add_argument("metadata", help="metadata.json 路径")
    parser.add_argument("--data-dir", required=True, help="数据目录（包含 rgb/, depth/）")
    parser.add_argument("-o", "--output", help="输出路径")
    parser.add_argument("-n", "--num-questions", type=int, default=5, help="生成题目数量")
    parser.add_argument("--max-images", type=int, default=6, help="最多发送多少张图像给 Claude")

    args = parser.parse_args()

    output_path = args.output or args.metadata.replace("metadata.json", "exam_v2.json")

    exam = generate_exam_v2(
        args.metadata,
        args.data_dir,
        output_path=output_path,
        num_questions=args.num_questions,
        max_images=args.max_images
    )

    print(f"\n✅ 考题生成完成！")
    print(f"   Exam ID: {exam.exam_id}")
    print(f"   Questions: {exam.total_questions}")


if __name__ == "__main__":
    main()

