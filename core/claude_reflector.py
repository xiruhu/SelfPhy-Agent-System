"""
claude_reflector.py
----------------------
三步排除法错题诊断系统 V2

输入：EvaluationResultV2（模型评测结果）+ ExamPaperV2（考卷含真实轨迹）
输出：每道错题的 ErrorAnalysisV2（证据链 + 根因分类）

三步排除法：
  Step 1: 物理-语义对齐检查  — 模型说的方向是否物理上可能？
  Step 2: 空间位置重塑验证  — 基于精确位姿，反算正确答案，量化偏差
  Step 3: 根因分类归纳      — 综合前两步，输出结构化根因

注：待测模型接收全部帧，FOV 可见性不再作为排除条件。

长官模型：claude-sonnet-4-6（只有它才接受多模态输入 + 执行推理）
"""

import json
import os
import sys
import base64
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum

import anthropic
from dotenv import load_dotenv

sys.path.append(str(Path(__file__).parent.parent))
from schema.question import QuestionV2, EvaluationResultV2

load_dotenv()

SUPERVISOR_MODEL = "claude-sonnet-4-6"
client = anthropic.Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    base_url=os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
)


class ErrorType(str, Enum):
    DIRECTION_CALC_ERROR    = "direction_calc_error"      # 坐标系变换计算错误
    ROTATION_SENSE_ERROR    = "rotation_sense_error"      # 旋转方向（顺/逆时针）判断反了
    ROTATION_TRANSLATION_CONFUSION = "rotation_translation_confusion"  # 旋转与平移混淆
    MEMORY_DECAY            = "memory_decay"              # 跨帧记忆丢失
    OBJECT_HALLUCINATION    = "object_hallucination"      # 幻觉物体
    FOV_MISUNDERSTANDING    = "fov_misunderstanding"      # 视场/遮挡理解错误


# 每种错误类型的诊断提示词
_ERROR_HINTS = {
    ErrorType.DIRECTION_CALC_ERROR:
        "模型感知到了旋转，但最终角度计算有误（如把-270°误算为+180°）",
    ErrorType.ROTATION_SENSE_ERROR:
        "模型把顺时针旋转看成了逆时针，或反之",
    ErrorType.ROTATION_TRANSLATION_CONFUSION:
        "模型把视角旋转（原地转身）产生的画面变化当成了位置移动",
    ErrorType.MEMORY_DECAY:
        "模型在中间帧丢失了对目标物体的空间记忆",
    ErrorType.OBJECT_HALLUCINATION:
        "模型描述了视频中并不存在的物体",
    ErrorType.FOV_MISUNDERSTANDING:
        "模型错误推断了物体的可见性或遮挡状态",
}


def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def analyze_one_error(
    question: Dict,
    response: Dict,
    exam_evidence: Dict,
    traj_data: List[Dict]
) -> Dict:
    """
    对单道错题执行三步排除法。

    Args:
        question:      exam_v2.json 中的题目字典
        response:      EvaluationResultV2 中对应的 response 字典
        exam_evidence: exam_v2.json 中的 multimodal_evidence
        traj_data:     完整关键帧位姿序列（来自 metadata.json）

    Returns:
        ErrorAnalysisV2 字典
    """
    q_id       = question["question_id"]
    capability = question["capability"]
    gt         = question["ground_truth_answer"]
    model_ans  = response["model_response"]
    score      = response.get("score", 0.0)

    # 取证据帧范围内的位姿数据
    eids = question.get("evidence_frame_ids", [])
    if eids:
        min_f, max_f = min(eids), max(eids)
        traj_window = [t for t in traj_data if min_f <= t["frame_id"] <= max_f]
    else:
        traj_window = traj_data

    # 格式化轨迹摘要（传给 Claude 的文本部分）
    traj_lines = "\n".join(
        f"  frame {t['frame_id']:>3}: yaw={t['yaw']:>8.2f}°  cum_yaw={t['cum_yaw_delta']:>8.2f}°  cum_disp={t['cum_displacement']:>6.3f}m"
        for t in traj_window
    )

    # 选首尾两帧图像作为视觉证据（避免 token 过多）
    image_blocks = []
    rgb_frames = exam_evidence.get("rgb_frames", {})
    selected_fids = []
    if eids:
        selected_fids = [eids[0], eids[len(eids)//2], eids[-1]]
    for fid in selected_fids:
        path = rgb_frames.get(str(fid))
        if path and Path(path).exists():
            image_blocks.append({
                "type": "image",
                "source": {"type": "base64", "media_type": "image/jpeg",
                           "data": encode_image(path)}
            })
            image_blocks.append({"type": "text", "text": f"[frame {fid}]"})

    prompt = f"""你是 SelfPhy-Agent-System 的错题诊断专家（Inspector Agent V2）。

## 题目信息
- 题目 ID: {q_id}
- 能力维度: {capability}
- 标准答案: {gt}
- 模型回答: {model_ans}
- 答题得分: {score:.2f}/1.00

## 证据帧图像（已附上首/中/尾帧）

## 精确位姿轨迹（证据帧范围内）
{traj_lines}

## 你的任务：执行三步排除法

**Step 1 - 物理-语义对齐检查**
模型的回答在语义上是否符合场景的物理约束？是否描述了不可能存在的物体或方向？

**Step 2 - 空间位置重塑验证**
根据精确位姿轨迹，计算正确答案：
- 起始 yaw = 轨迹第一帧 yaw
- 终止 yaw = 轨迹最后一帧 yaw
- 累计转向 cum_yaw_delta = 终止 - 起始
- 若物体初始在方向 θ₀，则终止时相对方向 = θ₀ - cum_yaw_delta（规范化到 [-180°, 180°]）
写出完整的数值推导过程，与模型回答对比，量化偏差。

**Step 3 - 根因分类**
从以下类型中选一个最匹配的根因：
{json.dumps({e.value: h for e, h in _ERROR_HINTS.items()}, ensure_ascii=False, indent=2)}

## 输出格式（严格 JSON，不加 markdown）
{{
  "step1_physical_check": {{
    "has_hallucination": false,
    "physical_violation": "描述或 null",
    "conclusion": "通过/发现问题"
  }},
  "step2_spatial_reconstruction": {{
    "start_yaw": 数值,
    "end_yaw": 数值,
    "cum_yaw_delta": 数值,
    "correct_answer_calc": "推导过程",
    "model_answer_deviation": "偏差描述"
  }},
  "step3_root_cause": {{
    "error_type": "错误类型枚举值",
    "confidence": 0到1,
    "explanation": "解释"
  }}
}}"""

    content = [{"type": "text", "text": prompt}] + image_blocks

    import time as _time
    import re as _re

    diagnosis = {"raw": ""}
    for _attempt in range(3):
        try:
            resp = client.messages.create(
                model=SUPERVISOR_MODEL,
                max_tokens=2000,
                messages=[{"role": "user", "content": content}]
            )
            # 用 .type == "text" 精确匹配 TextBlock，跳过 ThinkingBlock
            raw_block = next((b for b in resp.content if getattr(b, "type", None) == "text"), None)
            raw = raw_block.text.strip() if raw_block else ""

            if not raw:
                if _attempt < 2:
                    _time.sleep(3)
                    continue
                break

            # 解析 JSON（中文引号替换 + 括号匹配）
            raw = _re.sub(r"```[a-z]*\n?", "", raw)
            raw = _re.sub(r"```", "", raw).strip()
            raw_clean = raw.replace(""", '"').replace(""", '"').replace("'", "'").replace("'", "'")
            start = raw_clean.find('{')
            if start != -1:
                depth, end = 0, start
                for i, ch in enumerate(raw_clean[start:], start):
                    if ch == '{':
                        depth += 1
                    elif ch == '}':
                        depth -= 1
                        if depth == 0:
                            end = i
                            break
                try:
                    diagnosis = json.loads(raw_clean[start:end + 1])
                except json.JSONDecodeError:
                    diagnosis = {"raw": raw}
            else:
                diagnosis = {"raw": raw}
            break

        except Exception as e:
            diagnosis = {"error": str(e)}
            break

    root_cause = diagnosis.get("step3_root_cause", {})

    return {
        "question_id": q_id,
        "capability": capability,
        "model_name": response.get("model_name", "unknown"),
        "ground_truth": gt,
        "model_answer": model_ans,
        "score": score,
        "diagnosis": diagnosis,
        "error_type": root_cause.get("error_type", "unknown"),
        "confidence": root_cause.get("confidence", 0.0),
        "root_cause_explanation": root_cause.get("explanation", ""),
        "timestamp": datetime.now().isoformat()
    }


def run_reflector_v2(
    result_path: str,
    exam_path: str,
    metadata_path: str,
    output_path: str = None
) -> List[Dict]:
    """
    对评测结果中所有错题执行三步排除法诊断。

    Args:
        result_path:   test_result_kimi_v2_final.json
        exam_path:     test_exam_v2_precise.json
        metadata_path: metadata.json（含完整位姿序列）
        output_path:   输出路径

    Returns:
        错题诊断列表
    """
    from core.claude_examiner import compute_cumulative_trajectory

    with open(result_path, "r", encoding="utf-8") as f:
        result = json.load(f)
    with open(exam_path, "r", encoding="utf-8") as f:
        exam = json.load(f)
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    model_name = result["model_name"]
    questions   = {q["question_id"]: q for q in exam["questions"]}
    evidence    = exam["multimodal_evidence"]
    traj_data   = compute_cumulative_trajectory(metadata["keyframes"])

    # 只处理答错的题
    wrong_responses = [r for r in result["responses"] if not r["is_correct"]]
    print(f"\n[Reflector V2] 开始诊断 {len(wrong_responses)} 道错题（模型: {model_name}）")

    analyses = []
    for i, resp in enumerate(wrong_responses, 1):
        qid = resp["question_id"]
        q   = questions.get(qid)
        if q is None:
            continue
        print(f"  [{i}/{len(wrong_responses)}] {qid} ...")
        resp["model_name"] = model_name
        analysis = analyze_one_error(q, resp, evidence, traj_data)
        analyses.append(analysis)
        print(f"    根因: {analysis['error_type']} (置信度: {analysis['confidence']:.2f})")

    # 保存
    if output_path is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"outputs/reports/diagnosis_{model_name}_{ts}.json"

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(analyses, f, indent=2, ensure_ascii=False)

    print(f"\n✅ 诊断报告已保存: {output_path}")

    # 统计错误类型分布
    from collections import Counter
    type_dist = Counter(a["error_type"] for a in analyses)
    print("\n错误类型分布:")
    for etype, cnt in type_dist.most_common():
        print(f"  {etype}: {cnt} 道")

    return analyses


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Claude Reflector V2 - 三步排除法错题诊断")
    parser.add_argument("result",   help="EvaluationResultV2 JSON 路径")
    parser.add_argument("exam",     help="ExamPaperV2 JSON 路径")
    parser.add_argument("metadata", help="metadata.json 路径")
    parser.add_argument("-o", "--output", help="输出路径")
    args = parser.parse_args()
    run_reflector_v2(args.result, args.exam, args.metadata, args.output)


if __name__ == "__main__":
    main()
