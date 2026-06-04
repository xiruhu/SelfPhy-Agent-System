"""
claude_examiner.py
------------------
SelfPhy-Agent-System 的考试生成器（Supervisor Agent 模块）。

职责：
  - 读取 cv2_processor 输出的 metadata.json（含 yaw、motion_score、frame_name 等）
  - 对满足阈值的关键帧，调用 Claude Sonnet 4.6（长官模型）自动生成
    第一人称空间推理考题（Egocentric Spatial Recall）
  - 输出标准化 exam.json，供 evaluate_runner.py 派发给被测模型

数据流：
  outputs/metadata/metadata.json  →  [本模块]  →  outputs/exams/exam.json
"""

import json
import os
import sys
import time
from pathlib import Path

import anthropic
from dotenv import load_dotenv

# ─────────────────────────────────────────────
# 环境与路径配置
# ─────────────────────────────────────────────

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
ANTHROPIC_BASE_URL = os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com")

if not ANTHROPIC_API_KEY:
    sys.exit("[ERROR] 未找到 ANTHROPIC_API_KEY，请检查 .env 文件。")

# 长官模型：claude-sonnet-4-6（与 CLAUDE.md 架构设计一致）
SUPERVISOR_MODEL = "claude-sonnet-4-6"

METADATA_PATH = Path("outputs/metadata/metadata.json")
EXAM_OUTPUT   = Path("outputs/exams/exam_raw.json")
EXAM_OUTPUT.parent.mkdir(parents=True, exist_ok=True)

# 出题触发阈值（yaw 偏转角度 & 光流运动分数）
YAW_THRESHOLD    = 20    # 度，绝对值
MOTION_THRESHOLD = 15    # 光流运动分数

# ─────────────────────────────────────────────
# 初始化 Anthropic 客户端
# ─────────────────────────────────────────────

client = anthropic.Anthropic(
    api_key=ANTHROPIC_API_KEY,
    base_url=ANTHROPIC_BASE_URL,
)

# ─────────────────────────────────────────────
# Prompt 模板
# ─────────────────────────────────────────────

SYSTEM_PROMPT = """你是 SelfPhy-Agent-System 的考试生成器（Examiner Agent）。

你的任务是基于第一人称（Egocentric）视角的运动数据，生成高质量的空间推理考题，
用于评测大语言模型在动态物理世界中的空间记忆与推理能力。

【出题原则】
1. 所有题目必须以第一人称视角（"我"）描述场景。
2. 题目必须考察"转身/转向后物体方向变化"的空间记忆能力。
3. 答案必须是确定性的、可验证的（基于给定的运动参数）。
4. 题目难度应与 yaw 变化幅度和 motion_score 正相关。
5. 严格输出 JSON，不添加任何解释或 markdown 代码块。

【能力分类标准】
- Relative Direction Memory：转身后判断物体相对方向
- Occlusion Reasoning：物体被遮挡后的位置推断
- Spatial Backtracking：回溯路径中的空间关系
- Distance Estimation：运动距离与物体远近的估算"""


def build_user_prompt(frame_name: str, yaw: float, motion_score: float,
                      pitch: float = 0.0, roll: float = 0.0,
                      displacement_m: float = None) -> str:
    """
    构建给 Claude 的出题 Prompt。

    参数说明：
      frame_name     : 关键帧文件名，用于题目溯源
      yaw            : 水平偏转角（度），正值=右转，负值=左转
      motion_score   : 光流运动强度分数（越高=运动越剧烈）
      pitch          : 俯仰角（度），可选
      roll           : 横滚角（度），可选
      displacement_m : 估算位移（米），可选，来自 Habitat 6-DoF 数据
    """
    direction = "右转" if yaw > 0 else "左转"
    abs_yaw   = abs(yaw)

    # 难度标签
    if abs_yaw < 45:
        difficulty = "简单"
    elif abs_yaw < 90:
        difficulty = "中等"
    else:
        difficulty = "困难"

    displacement_info = ""
    if displacement_m is not None:
        displacement_info = f"\n- 估算位移: {displacement_m:.2f} 米"

    return f"""请根据以下第一人称运动数据，生成一道空间推理考题。

【运动参数】
- 关键帧: {frame_name}
- 偏转方向: {direction} {abs_yaw:.1f}°（yaw = {yaw:.1f}）
- 运动强度: {motion_score:.1f}（motion_score）
- 俯仰角: {pitch:.1f}°（pitch）
- 横滚角: {roll:.1f}°（roll）{displacement_info}
- 预期难度: {difficulty}

【出题要求】
1. 场景设定：在一个室内环境中，我正在行走并观察周围物体。
2. 题目类型：优先选择 Relative Direction Memory（相对方向记忆）。
   若 motion_score > 30，可考虑 Occlusion Reasoning（遮挡推理）。
3. 题目必须包含：
   - 转向前看到的物体（至少 2 个，位于不同方向）
   - 转向动作描述（与 yaw 参数一致）
   - 转向后的方向判断问题
4. 答案必须基于几何推导，给出明确方向（前/后/左/右/左前/右后等）。

【输出格式】严格输出以下 JSON，不添加任何其他内容：
{{
    "question": "（第一人称场景描述 + 问题，100-200字）",
    "answer": "（明确的方向答案 + 简短推导，50字以内）",
    "capability": "（Relative Direction Memory / Occlusion Reasoning / Spatial Backtracking / Distance Estimation 之一）",
    "difficulty": "（简单 / 中等 / 困难）",
    "reasoning_chain": "（出题时的几何推导过程，用于验证答案正确性，100字以内）"
}}"""


# ─────────────────────────────────────────────
# 核心出题函数
# ─────────────────────────────────────────────

def generate_exam_item(frame_name: str, yaw: float, motion_score: float,
                       pitch: float = 0.0, roll: float = 0.0,
                       displacement_m: float = None,
                       image_path: str = "",
                       max_retries: int = 3) -> dict | None:
    """
    调用 Claude Sonnet 4.6 为单个关键帧生成考题。

    返回包含题目信息的 dict，失败时返回 None。
    """
    user_prompt = build_user_prompt(
        frame_name, yaw, motion_score, pitch, roll, displacement_m
    )

    for attempt in range(1, max_retries + 1):
        try:
            response = client.messages.create(
                model=SUPERVISOR_MODEL,
                max_tokens=1024,
                thinking={"type": "adaptive"},   # 自适应思考，提升推理质量
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
            )

            # 提取文本内容（跳过 thinking block）
            raw_text = ""
            for block in response.content:
                if block.type == "text":
                    raw_text = block.text.strip()
                    break

            if not raw_text:
                print(f"  [WARN] {frame_name}: 响应为空，重试 {attempt}/{max_retries}")
                continue

            # 清理可能的 markdown 代码块包裹
            if raw_text.startswith("```"):
                lines = raw_text.split("\n")
                raw_text = "\n".join(lines[1:-1]) if lines[-1] == "```" else "\n".join(lines[1:])

            exam_item = json.loads(raw_text)

            # 注入元数据
            exam_item["frame_name"]    = frame_name
            exam_item["yaw"]           = yaw
            exam_item["motion_score"]  = motion_score
            exam_item["pitch"]         = pitch
            exam_item["roll"]          = roll
            if displacement_m is not None:
                exam_item["displacement_m"] = displacement_m
            # 透传图片路径，供 exam_formatter 分流使用
            exam_item["image_path"]    = image_path

            # 校验必要字段
            required_fields = {"question", "answer", "capability"}
            if not required_fields.issubset(exam_item.keys()):
                missing = required_fields - exam_item.keys()
                print(f"  [WARN] {frame_name}: 缺少字段 {missing}，重试 {attempt}/{max_retries}")
                continue

            return exam_item

        except json.JSONDecodeError as e:
            print(f"  [WARN] {frame_name}: JSON 解析失败（{e}），重试 {attempt}/{max_retries}")
            print(f"         原始响应: {raw_text[:200]}")
        except anthropic.RateLimitError:
            wait = 60 * attempt
            print(f"  [RATE LIMIT] 等待 {wait}s 后重试...")
            time.sleep(wait)
        except anthropic.APIStatusError as e:
            print(f"  [API ERROR] {frame_name}: {e.status_code} - {e.message}")
            if e.status_code < 500:
                return None  # 客户端错误不重试
        except Exception as e:
            print(f"  [ERROR] {frame_name}: 未知错误 {e}")

    print(f"  [FAIL] {frame_name}: 达到最大重试次数，跳过。")
    return None


# ─────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────

def main():
    print(f"[SelfPhy Examiner] 使用模型: {SUPERVISOR_MODEL}")
    print(f"[SelfPhy Examiner] 读取 metadata: {METADATA_PATH}")

    if not METADATA_PATH.exists():
        sys.exit(f"[ERROR] 找不到 metadata 文件: {METADATA_PATH}")

    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata: list[dict] = json.load(f)

    print(f"[SelfPhy Examiner] 共 {len(metadata)} 帧，筛选阈值: |yaw| > {YAW_THRESHOLD}° & motion_score > {MOTION_THRESHOLD}")

    exam_list: list[dict] = []
    skipped = 0

    for item in metadata:
        yaw          = item.get("yaw", 0.0)
        motion_score = item.get("motion_score", 0.0)
        frame_name   = item.get("frame_name", "unknown")

        # 阈值过滤：只对运动显著的关键帧出题
        if abs(yaw) <= YAW_THRESHOLD or motion_score <= MOTION_THRESHOLD:
            skipped += 1
            continue

        print(f"  [出题] {frame_name}  yaw={yaw:.1f}°  motion={motion_score:.1f}")

        exam_item = generate_exam_item(
            frame_name   = frame_name,
            yaw          = yaw,
            motion_score = motion_score,
            pitch        = item.get("pitch", 0.0),
            roll         = item.get("roll", 0.0),
            displacement_m = item.get("displacement_m"),
            image_path   = item.get("image_path", ""),
        )

        if exam_item:
            exam_list.append(exam_item)
            print(f"    ✓ 能力类型: {exam_item.get('capability')}  难度: {exam_item.get('difficulty', 'N/A')}")
        else:
            print(f"    ✗ 出题失败，已跳过")

    # 保存结果
    with open(EXAM_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(exam_list, f, ensure_ascii=False, indent=4)

    print(f"\n[SelfPhy Examiner] 完成！")
    print(f"  总帧数: {len(metadata)}")
    print(f"  跳过（未达阈值）: {skipped}")
    print(f"  成功出题: {len(exam_list)}")
    print(f"  输出路径: {EXAM_OUTPUT}")


if __name__ == "__main__":
    main()
