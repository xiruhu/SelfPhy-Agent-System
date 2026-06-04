"""
exam_formatter.py
-----------------
试卷分流模块：将 claude_examiner 输出的 exam_raw.json 转换为
各被测模型专用的标准化试卷格式，供 evaluate_runner.py 直接消费。

数据流：
  outputs/exams/exam_raw.json  →  [本模块]
    ├─ outputs/exams/exam_kimi.json      (文本 + base64 图片，OpenAI 兼容)
    ├─ outputs/exams/exam_minimax.json   (纯文本，无图片)
    └─ outputs/exams/exam_doubao.json    (文本 + base64 图片，OpenAI 兼容)

统一 Question 格式（evaluate_runner 消费）：
  {
    "question_id":     "Q001_kimi",
    "model_target":    "kimi",
    "prompt":          "（发给模型的完整文本提示）",
    "context_frames":  ["path/to/frame.jpg"],   # 纯文本模型为空列表
    "supports_vision": true/false,
    "ground_truth": {
        "answer":          "（标准答案）",
        "capability":      "Relative Direction Memory",
        "difficulty":      "中等",
        "reasoning_chain": "（几何推导，用于 inspector_agent）"
    },
    "raw_exam": { ... }   # 原始 exam_raw 条目，供调试
  }
"""

import json
import sys
from pathlib import Path


EXAM_RAW_PATH = Path("outputs/exams/exam_raw.json")
EXAM_DIR      = Path("outputs/exams")
EXAM_DIR.mkdir(parents=True, exist_ok=True)

# 各模型能力声明
MODEL_CAPS = {
    "kimi":    {"supports_vision": True,  "max_frames": 4},
    "minimax": {"supports_vision": False, "max_frames": 0},
    "doubao":  {"supports_vision": True,  "max_frames": 4},
}


# ─────────────────────────────────────────────
# Prompt 构建
# ─────────────────────────────────────────────

def build_vision_prompt(item: dict) -> str:
    """
    为支持视觉的模型构建 prompt。
    图片通过 context_frames 字段传递，prompt 里只做文字引导。
    """
    direction = "右转" if item["yaw"] > 0 else "左转"
    abs_yaw   = abs(item["yaw"])

    return f"""你正在参加一项第一人称空间推理测试。

【场景信息】
- 你刚刚发生了一次视角转向：{direction} {abs_yaw:.1f}°
- 运动强度评分：{item['motion_score']:.1f}
- 上方图片是该转向时刻的第一人称视角关键帧

【考题】
{item['question']}

【作答要求】
请给出明确的方向答案（如：左方、右前方、正后方等），并简要说明你的推理过程（2-3句话）。
不需要重复题目内容，直接给出答案。"""


def build_text_only_prompt(item: dict) -> str:
    """
    为不支持视觉的模型（MiniMax）构建纯文本 prompt。
    将视觉信息转化为文字描述，确保题目可解。
    """
    direction = "右转" if item["yaw"] > 0 else "左转"
    abs_yaw   = abs(item["yaw"])

    # 难度映射
    difficulty_map = {"简单": "easy", "中等": "medium", "困难": "hard"}
    difficulty_en  = difficulty_map.get(item.get("difficulty", "中等"), "medium")

    return f"""你正在参加一项第一人称空间推理测试（纯文字模式）。

【运动描述】
- 转向方向：{direction} {abs_yaw:.1f}°
- 运动强度：{item['motion_score']:.1f}（数值越高代表运动越剧烈）
- 位移距离：{item.get('displacement_m', 0):.2f} 米
- 难度等级：{item.get('difficulty', '中等')}（{difficulty_en}）

【考题】
{item['question']}

【作答要求】
请给出明确的方向答案（如：左方、右前方、正后方等），并简要说明你的推理过程（2-3句话）。
注意：本题基于纯文字描述，请根据转向角度和运动参数进行几何推理。"""


# ─────────────────────────────────────────────
# 格式化函数
# ─────────────────────────────────────────────

def format_for_model(raw_items: list[dict], model_name: str) -> list[dict]:
    """将 exam_raw 条目转换为指定模型的标准化试卷条目。"""
    caps    = MODEL_CAPS[model_name]
    results = []

    for idx, item in enumerate(raw_items):
        question_id = f"Q{idx+1:03d}_{model_name}"

        # 构建 prompt
        if caps["supports_vision"]:
            prompt = build_vision_prompt(item)
            # 只取一帧（关键帧），路径可能是相对路径
            image_path = item.get("image_path", "")
            context_frames = [image_path] if image_path and Path(image_path).exists() else []
        else:
            prompt = build_text_only_prompt(item)
            context_frames = []

        results.append({
            "question_id":     question_id,
            "model_target":    model_name,
            "prompt":          prompt,
            "context_frames":  context_frames,
            "supports_vision": caps["supports_vision"],
            "ground_truth": {
                "answer":          item.get("answer", ""),
                "capability":      item.get("capability", ""),
                "difficulty":      item.get("difficulty", ""),
                "reasoning_chain": item.get("reasoning_chain", ""),
            },
            "raw_exam": {
                "frame_name":   item.get("frame_name"),
                "yaw":          item.get("yaw"),
                "motion_score": item.get("motion_score"),
                "question":     item.get("question"),
            }
        })

    return results


# ─────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────

def main():
    if not EXAM_RAW_PATH.exists():
        print(f"[ERROR] 找不到原始试卷：{EXAM_RAW_PATH}")
        print("请先运行 claude_examiner.py 生成 exam_raw.json。")
        sys.exit(1)

    with open(EXAM_RAW_PATH, "r", encoding="utf-8") as f:
        raw_items: list[dict] = json.load(f)

    if not raw_items:
        print("[WARN] exam_raw.json 为空，无题目可分流。")
        sys.exit(0)

    print(f"[exam_formatter] 读取 {len(raw_items)} 道题，开始分流...")

    for model_name in MODEL_CAPS:
        formatted = format_for_model(raw_items, model_name)
        out_path  = EXAM_DIR / f"exam_{model_name}.json"

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(formatted, f, ensure_ascii=False, indent=2)

        vision_note = "（含图片）" if MODEL_CAPS[model_name]["supports_vision"] else "（纯文本）"
        print(f"  ✓ {model_name:10s} {vision_note}  → {out_path}  ({len(formatted)} 题)")

    print("\n[exam_formatter] 分流完成。")
    print("下一步：运行 evaluate_runner.py --model kimi/minimax/doubao")


if __name__ == "__main__":
    main()
