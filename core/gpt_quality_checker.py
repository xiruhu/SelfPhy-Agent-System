"""
gpt_quality_checker.py
-----------------------
GPT-4o 考题质量审查器（Supervisor Agent 2）

职责：对 Claude 生成的每道考题，结合关键帧图像和轨迹数据，审查：
1. 答案数学正确性：独立用位姿数据验算 ground_truth_answer
2. 题干空间推理有效性：该题是否真正要求被测模型理解第一人称视觉运动

数据流：
  exam_v2.json + 关键帧图像 (来自 claude_examiner.py)
        ↓
  [本模块] GPT-4o 逐题审核（图像 + 轨迹 + 题目）
        ↓
  exam_v2_refined.json → evaluate_runner.py
"""

import base64
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

from openai import OpenAI
from dotenv import load_dotenv

sys.path.append(str(Path(__file__).parent.parent))
from core.claude_examiner import compute_cumulative_trajectory

_project_root = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=_project_root / ".env", override=True)

CHECKER_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

_client: Optional[OpenAI] = None

def _get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        if not api_key:
            sys.exit("[ERROR] 未找到 OPENAI_API_KEY，请检查 .env 文件。")
        _client = OpenAI(api_key=api_key, base_url=base_url)
    return _client


# ─────────────────────────────────────────────
# System Prompt
# ─────────────────────────────────────────────

CHECKER_SYSTEM_PROMPT = """你是 SelfPhy-Agent-System 的考题质量审查员。

【系统背景】
这是一个评测大模型第一人称空间推理能力的考官系统。
- Claude（出题模型）：拿到精确的位姿轨迹数据和关键帧，生成考题与标准答案
- 被测模型（Kimi、豆包等）：只收到全部视频帧和题目，必须通过"看图理解运动"来作答
- 你（GPT-4o）：审查 Claude 出的题，把关两件事

【你的两项审查任务】

## 任务 1：答案数学正确性
独立验算 ground_truth_answer，不要信任 Claude 给的 reasoning_trace，自己算：

空间变换公式：
  物体最终相对方向 = 物体初始相对方向 - 累计转向(cum_yaw_delta)
  结果归一化到 (-180°, 180°]（若超出则 ±360°）

方位词映射（±22.5° 容差）：
  0° = 正前方 | ±90° = 正左/右方 | ±180° = 正后方
  ±45° = 左/右前方 | ±135° = 左/右后方

distance_estimation 类题：答案应 ≈ 证据帧范围内的 cum_displacement（±20% 误差）

若计算结果与 ground_truth_answer 不符 → action = fix_answer，给出正确答案和推导

## 任务 2：题干空间推理有效性
结合你看到的图像序列，判断这道题是否真正要求被测模型：
  (a) 从图像中感知自身运动（旋转/平移方向和幅度）
  (b) 建立跨帧的空间记忆（记住某物体的初始位置）
  (c) 将两者结合推理出答案

判为无效的情形：
  - 仅凭文字逻辑就能回答（无需看图）
  - 图像中根本看不到题干指代的物体（第0帧里根本没有可见的参照物）
  - 题目考察的运动区间在图像中变化极小、无法感知

判为有效的情形：
  - 需要看第0帧识别参照物，再看后续帧感知运动，才能推断答案
  - 图像中有明确的可辨认物体作为参照

【操作决策】
- keep：两项检查均通过
- fix_answer：题干有效，但答案计算错误，给出正确答案
- delete：题目有根本性缺陷（答案无法验算 且 无法修正，或题干根本不考察空间推理且无法修复）
- rewrite：题干存在可修复的空间推理有效性问题（如参照物在图中不可见，改用更明显的参照描述）

【输出格式】
严格输出 JSON 对象（单题），不加任何 markdown：
{
  "question_id": "...",
  "action": "keep|fix_answer|delete|rewrite",
  "answer_check": {
    "my_calculation": "推导过程",
    "my_result": "计算得到的正确答案",
    "is_correct": true/false,
    "deviation": "偏差描述或 null"
  },
  "reasoning_check": {
    "requires_visual_motion": true/false,
    "reference_object_visible": true/false,
    "verdict": "有效/无效",
    "reason": "判断依据"
  },
  "fixed_answer": "修正后的答案（仅 fix_answer 时填，否则 null）",
  "fixed_reasoning_trace": "修正后的推导（仅 fix_answer 时填，否则 null）",
  "rewritten_question": "优化后的题干（仅 rewrite 时填，否则 null）",
  "quality_score": 0.0到1.0
}"""


# ─────────────────────────────────────────────
# 辅助函数
# ─────────────────────────────────────────────

def _encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _build_question_message(
    question: Dict,
    traj_data: List[Dict],
    keyframe_images: List[Dict],  # 与 Claude 出题时相同的关键帧列表
    data_base: Path,
) -> List[Dict]:
    """
    构建单题的 multimodal message content。
    图像使用与 Claude 出题时完全相同的关键帧（来自 metadata keyframes）。
    轨迹窗口限定在该题的 evidence_frame_ids 范围内。
    """
    eids = question.get("evidence_frame_ids", [])

    if eids:
        min_f, max_f = min(eids), max(eids)
        traj_window = [t for t in traj_data if min_f <= t["frame_id"] <= max_f]
    else:
        traj_window = traj_data

    traj_lines = "\n".join(
        f"  frame {t['frame_id']:>3}: cum_yaw={t['cum_yaw_delta']:>8.2f}°  "
        f"cum_disp={t['cum_displacement']:>6.3f}m  Δyaw={t['delta_yaw']:>7.2f}°"
        for t in traj_window
    )

    intro_text = (
        f"## 考题信息\n"
        f"题目 ID: {question['question_id']}\n"
        f"能力维度: {question['capability']}\n"
        f"难度: {question['difficulty']}\n"
        f"题干: {question['question_text']}\n"
        f"Claude 给出的标准答案: {question['ground_truth_answer']}\n"
        f"Claude 给出的推理过程: {question['reasoning_trace']}\n"
        f"rotation_degree: {question.get('rotation_degree', 'N/A')}°  "
        f"displacement_meters: {question.get('displacement_meters', 'N/A')}m\n\n"
        f"## 精确位姿轨迹（证据帧范围内，共 {len(traj_window)} 帧）\n"
        f"格式: frame | cum_yaw（累计转向°）| cum_disp（累计位移m）| Δyaw（与上帧差°）\n"
        f"{traj_lines}\n\n"
        f"## 关键帧图像（与 Claude 出题时相同的 {len(keyframe_images)} 张，按时间顺序）"
    )

    content = [{"type": "text", "text": intro_text}]

    for img in keyframe_images:
        fid = img["frame_id"]
        rgb_path = img.get("rgb_path")
        if not rgb_path:
            continue
        abs_path = Path(rgb_path) if Path(rgb_path).is_absolute() else data_base / rgb_path
        if not abs_path.exists():
            continue
        content.append({"type": "text", "text": f"[frame {fid}]"})
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{_encode_image(str(abs_path))}",
                "detail": "low"
            }
        })

    content.append({"type": "text", "text": "\n请完成两项审查并输出 JSON。"})
    return content


def _parse_single_response(raw: str, question_id: str) -> Dict:
    """解析单题 GPT 响应，容错处理。"""
    raw = re.sub(r'```[a-z]*\n?', '', raw).strip()
    raw = re.sub(r'```', '', raw).strip()

    # 找第一个完整 JSON 对象
    start = raw.find('{')
    if start == -1:
        return {"question_id": question_id, "action": "keep", "quality_score": 1.0, "reasoning": "parse_failed"}

    depth, end = 0, start
    for i, ch in enumerate(raw[start:], start):
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                end = i
                break

    try:
        result = json.loads(raw[start:end + 1])
        result.setdefault("question_id", question_id)
        return result
    except json.JSONDecodeError:
        return {"question_id": question_id, "action": "keep", "quality_score": 1.0, "reasoning": "parse_failed"}


# ─────────────────────────────────────────────
# 核心函数
# ─────────────────────────────────────────────

def check_exam_quality(
    exam_path: str,
    metadata_path: str,
    output_path: str = None,
    max_frames_per_question: int = 6
) -> Dict:
    """
    使用 GPT-4o 逐题审查 Claude 生成的考卷质量。
    GPT 看到与 Claude 出题时完全相同的关键帧图像。

    Args:
        exam_path:                exam_v2.json 路径
        metadata_path:            metadata.json 路径
        output_path:              输出路径（精炼后的 exam JSON）
        max_frames_per_question:  保留参数，本版本不再截断（与 Claude 保持一致）

    Returns:
        审核后的考卷字典（兼容 exam_v2 格式）
    """
    with open(exam_path, 'r', encoding='utf-8') as f:
        exam = json.load(f)
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    exam_id = exam["exam_id"]
    questions = exam["questions"]

    # 从 metadata 重建与 Claude 出题时相同的关键帧列表
    # 对应 claude_examiner.load_keyframe_images 的逻辑：均匀采样 max_images=6 帧
    keyframes = metadata["keyframes"]
    max_images = 6
    if len(keyframes) > max_images:
        indices = [int(i * len(keyframes) / max_images) for i in range(max_images)]
    else:
        indices = list(range(len(keyframes)))

    data_dir = Path(metadata_path).parent
    keyframe_images = []
    for idx in indices:
        kf = keyframes[idx]
        fid = kf["frame_id"]
        rgb_rel = kf.get("rgb")
        if rgb_rel:
            rgb_abs = data_dir / rgb_rel
            keyframe_images.append({
                "frame_id": fid,
                "rgb_path": str(rgb_abs) if rgb_abs.exists() else None
            })

    print(f"\n[GPT Quality Checker] 开始审核考卷: {exam_id}")
    print(f"  审核模型: {CHECKER_MODEL}")
    print(f"  题目数量: {len(questions)}  关键帧: {len(keyframe_images)} 张（与 Claude 出题时相同）")

    traj_data = compute_cumulative_trajectory(keyframes)

    # ── 逐题审核 ──
    check_results = []
    for i, q in enumerate(questions, 1):
        qid = q["question_id"]
        print(f"  [{i}/{len(questions)}] {qid} ...", end=" ", flush=True)

        content = _build_question_message(q, traj_data, keyframe_images, _project_root)

        try:
            t0 = time.time()
            response = _get_client().chat.completions.create(
                model=CHECKER_MODEL,
                messages=[
                    {"role": "system", "content": CHECKER_SYSTEM_PROMPT},
                    {"role": "user", "content": content}
                ],
                temperature=0.1,
                max_tokens=1200,
                timeout=120
            )
            elapsed = time.time() - t0
            raw = response.choices[0].message.content.strip()
            result = _parse_single_response(raw, qid)
            print(f"action={result.get('action', '?')}  ({elapsed:.1f}s)")
        except Exception as e:
            print(f"ERROR: {e}")
            result = {"question_id": qid, "action": "keep", "quality_score": 1.0, "reasoning": f"api_error: {e}"}

        check_results.append(result)

    # ── 应用审查结果 ──
    result_map = {r["question_id"]: r for r in check_results}
    refined_questions = []
    stats = {"deleted": 0, "fixed": 0, "rewritten": 0, "kept": 0}

    for q in questions:
        qid = q["question_id"]
        check = result_map.get(qid, {"action": "keep"})
        action = check.get("action", "keep")

        if action == "delete":
            stats["deleted"] += 1
            rc = check.get("reasoning_check", {})
            ac = check.get("answer_check", {})
            reason = rc.get("reason") or ac.get("deviation") or "根本性错误"
            print(f"  [DELETE]  {qid}: {reason}")
            continue

        q = dict(q)

        if action == "fix_answer":
            stats["fixed"] += 1
            old_answer = q["ground_truth_answer"]
            new_answer = check.get("fixed_answer") or old_answer
            q["ground_truth_answer"] = new_answer
            if check.get("fixed_reasoning_trace"):
                q["reasoning_trace"] = check["fixed_reasoning_trace"]
            q["quality_check"] = {
                "action": "fix_answer",
                "original_answer": old_answer,
                "answer_check": check.get("answer_check", {}),
                "quality_score": check.get("quality_score", 0.6)
            }
            print(f"  [FIX]     {qid}: '{old_answer}' → '{new_answer}'")

        elif action == "rewrite":
            stats["rewritten"] += 1
            old_text = q["question_text"]
            new_text = check.get("rewritten_question") or old_text
            q["question_text"] = new_text
            q["quality_check"] = {
                "action": "rewrite",
                "original_question": old_text,
                "reasoning_check": check.get("reasoning_check", {}),
                "quality_score": check.get("quality_score", 0.7)
            }
            print(f"  [REWRITE] {qid}: {check.get('reasoning_check', {}).get('reason', '')[:60]}")

        else:  # keep
            stats["kept"] += 1
            q["quality_check"] = {
                "action": "keep",
                "answer_check": check.get("answer_check", {}),
                "reasoning_check": check.get("reasoning_check", {}),
                "quality_score": check.get("quality_score", 1.0)
            }

        refined_questions.append(q)

    # ── 构建精炼考卷 ──
    refined_exam = dict(exam)
    refined_exam["questions"] = refined_questions
    refined_exam["total_questions"] = len(refined_questions)

    difficulty_dist: Dict[str, int] = {}
    capability_dist: Dict[str, int] = {}
    for q in refined_questions:
        difficulty_dist[q["difficulty"]] = difficulty_dist.get(q["difficulty"], 0) + 1
        capability_dist[q["capability"]] = capability_dist.get(q["capability"], 0) + 1
    refined_exam["difficulty_distribution"] = difficulty_dist
    refined_exam["capability_distribution"] = capability_dist
    refined_exam["quality_check_meta"] = {
        "checker_model": CHECKER_MODEL,
        "checked_at": datetime.now().isoformat(),
        "original_count": len(questions),
        "refined_count": len(refined_questions),
        **stats
    }

    print(f"\n[GPT Quality Checker] 审核完成:")
    print(f"  原始题数: {len(questions)}  →  保留: {len(refined_questions)}")
    print(f"  删除: {stats['deleted']}  更正答案: {stats['fixed']}  优化题干: {stats['rewritten']}  无改动: {stats['kept']}")

    if output_path is None:
        p = Path(exam_path)
        output_path = str(p.parent / p.name.replace("exam_", "exam_refined_"))

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(refined_exam, f, indent=2, ensure_ascii=False)

    print(f"\n✅ 精炼考卷已保存: {output_path}")
    return refined_exam


def main():
    import argparse
    parser = argparse.ArgumentParser(description="GPT-4o 考题质量审查器")
    parser.add_argument("exam", help="exam_v2.json 路径")
    parser.add_argument("metadata", help="metadata.json 路径")
    parser.add_argument("-o", "--output", help="输出路径（默认同目录 exam_refined_*.json）")
    parser.add_argument("--max-frames", type=int, default=6, help="每题最多发送的图像帧数")
    args = parser.parse_args()
    check_exam_quality(args.exam, args.metadata, args.output, args.max_frames)


if __name__ == "__main__":
    main()
