"""
evaluate_runner_v2.py
---------------------
模型测试器 V2：向被测模型发送纯问题 + 多模态证据

核心变化：
1. 不发送 Claude 的场景描述
2. 发送完整的多帧图像 + 轨迹数据 + 深度信息
3. 被测模型必须自己理解第一人称视觉运动

数据流：
  exam_v2.json
        ↓
  [本模块] 准备多模态输入
        ↓
  Kimi / MiniMax / 豆包
        ↓
  evaluation_result_v2.json
"""

import json
import os
import sys
import base64
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

from dotenv import load_dotenv

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from schema.question_v2 import (
    ExamPaperV2,
    QuestionV2,
    ModelInputV2,
    EvaluationResultV2
)

load_dotenv()

# ─────────────────────────────────────────────
# 答案判断模块
# ─────────────────────────────────────────────

# 方向同义词映射（从长到短排列，避免短词先命中）
_DIRECTION_SYNONYMS = {
    "正前方": ["正前方", "前方", "正前", "front", "forward", "ahead", "0°", "0度"],
    "正后方": ["正后方", "后方", "正后", "back", "behind", "180°", "180度", "-180°"],
    "正左方": ["正左方", "左方", "正左", "左边", "左侧", "left", "-90°", "-90度", "270°"],
    "正右方": ["正右方", "右方", "正右", "右边", "右侧", "right", "+90°", "90度", "90°"],
    "左前方": ["左前方", "前左方", "左前", "front-left", "left-front", "-45°", "-45度", "315°"],
    "右前方": ["右前方", "前右方", "右前", "front-right", "right-front", "+45°", "45度", "45°"],
    "左后方": ["左后方", "后左方", "左后", "back-left", "left-back", "-135°", "-135度", "225°"],
    "右后方": ["右后方", "后右方", "右后", "back-right", "right-back", "+135°", "135度", "135°"],
}

# 按同义词长度从长到短排序，保证长词优先命中
_SORTED_SYNONYMS = {
    canonical: sorted(syns, key=len, reverse=True)
    for canonical, syns in _DIRECTION_SYNONYMS.items()
}

def _normalize_direction(text: str) -> str:
    """
    把方向文本归一化为标准标签。
    对文本中出现的所有方向词取最长匹配，返回对应的标准标签。
    """
    text_lower = text.lower().strip()
    best_match = None
    best_len = 0
    for canonical, synonyms in _SORTED_SYNONYMS.items():
        for syn in synonyms:
            syn_lower = syn.lower()
            if syn_lower in text_lower and len(syn_lower) > best_len:
                best_match = canonical
                best_len = len(syn_lower)
    return best_match if best_match else text_lower


def _judge_direction(model_answer: str, ground_truth: str) -> tuple:
    """
    判断方向类答案是否正确。
    先做同义词归一化，再做包含匹配。
    Returns: (is_correct, score, detail)
    """
    norm_model = _normalize_direction(model_answer)
    norm_gt = _normalize_direction(ground_truth)

    if norm_model == norm_gt:
        return True, 1.0, f"归一化后完全匹配: '{norm_gt}'"

    # 归一化后做包含匹配（处理"右后方" vs "后方偏右"等描述）
    if norm_gt in norm_model or norm_model in norm_gt:
        return True, 0.9, f"归一化后包含匹配: model='{norm_model}' gt='{norm_gt}'"

    # 关键词部分匹配（方向词相同但描述不同，给部分分）
    gt_keywords = set(norm_gt.replace("方", "").replace("正", ""))
    model_keywords = set(norm_model.replace("方", "").replace("正", ""))
    overlap = gt_keywords & model_keywords
    if overlap and len(overlap) >= len(gt_keywords) * 0.5:
        return False, 0.5, f"部分匹配（关键词重叠 {overlap}），判为错误"

    return False, 0.0, f"不匹配: model='{norm_model}' gt='{norm_gt}'"


def _judge_numeric(model_answer: str, ground_truth: str, tolerance_ratio: float = 0.2) -> tuple:
    """
    判断数值类答案（distance_estimation）。
    允许 tolerance_ratio（默认20%）的误差范围。
    Returns: (is_correct, score, detail)
    """
    import re

    def extract_number(text: str) -> float:
        # 提取文本中的第一个数值
        nums = re.findall(r"\d+\.?\d*", text.replace(",", ""))
        return float(nums[0]) if nums else None

    gt_val = extract_number(ground_truth)
    model_val = extract_number(model_answer)

    if gt_val is None:
        return False, 0.0, "无法解析标准答案数值"
    if model_val is None:
        return False, 0.0, "模型回答中未找到数值"

    error_ratio = abs(model_val - gt_val) / max(gt_val, 1e-6)
    score = max(0.0, 1.0 - error_ratio / tolerance_ratio)

    if error_ratio <= tolerance_ratio:
        return True, score, f"数值在误差范围内: model={model_val}, gt={gt_val}, 误差={error_ratio:.1%}"
    else:
        return False, score, f"数值超出误差范围: model={model_val}, gt={gt_val}, 误差={error_ratio:.1%} > {tolerance_ratio:.0%}"


def judge_answer(
    model_answer: str,
    ground_truth: str,
    capability: str
) -> tuple:
    """
    根据题目类型选择判断策略。

    Args:
        model_answer: 模型的回答文本
        ground_truth: 标准答案文本
        capability: 题目能力类型

    Returns:
        (is_correct: bool, score: float 0-1, detail: str)
    """
    if not model_answer:
        return False, 0.0, "模型回答为空（可能 API 超时或出错）"

    if capability == "distance_estimation":
        return _judge_numeric(model_answer, ground_truth, tolerance_ratio=0.2)
    else:
        # 方向类题目：egocentric_memory, spatial_transformation,
        # occlusion_reasoning, trajectory_backtracking
        return _judge_direction(model_answer, ground_truth)

# ─────────────────────────────────────────────
# 辅助函数
# ─────────────────────────────────────────────

def encode_image_to_base64(image_path: str) -> str:
    """将图像编码为 base64"""
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


def prepare_multimodal_input_v2(
    question: QuestionV2,
    evidence: Dict
) -> ModelInputV2:
    """
    准备被测模型的输入。

    设计原则：
    - 传入 evidence_frame_ids 范围内的所有关键帧，不截断
    - 不传轨迹数值（position/rotation），只传图像
    - 待测模型只能通过"看视频"（连续帧序列）来推断空间关系
    """
    # 取 evidence_frame_ids 覆盖范围内的所有关键帧，按 frame_id 排序
    if question.evidence_frame_ids:
        min_fid = min(question.evidence_frame_ids)
        max_fid = max(question.evidence_frame_ids)
    else:
        min_fid, max_fid = 0, float('inf')

    candidate = sorted(
        [(int(k), v) for k, v in evidence["rgb_frames"].items()
         if min_fid <= int(k) <= max_fid],
        key=lambda x: x[0]
    )

    frames = []
    for frame_id, rgb_path in candidate:
        if rgb_path and Path(rgb_path).exists():
            frames.append({
                "frame_id": frame_id,
                "rgb": encode_image_to_base64(rgb_path),
                "depth": None
            })

    return ModelInputV2(
        question_id=question.question_id,
        question_text=question.question_text,
        frames=frames,
        trajectory=[],   # 故意置空
        has_depth=False
    )


# ─────────────────────────────────────────────
# 被测模型 API 调用
# ─────────────────────────────────────────────

def call_kimi_api_v2(model_input: ModelInputV2) -> Dict[str, Any]:
    """
    调用 Kimi API（支持多模态）

    Args:
        model_input: ModelInputV2 对象

    Returns:
        响应字典
    """
    import openai

    api_key = os.getenv("MOONSHOT_API_KEY")
    base_url = os.getenv("MOONSHOT_BASE_URL")
    model_name = os.getenv("MOONSHOT_MODEL", "kimi-k2.5")

    if not api_key:
        raise ValueError("未找到 MOONSHOT_API_KEY")

    client = openai.OpenAI(api_key=api_key, base_url=base_url)

    # 构建消息内容：问题 + 连续帧（模拟视频）
    message_content = [
        {
            "type": "text",
            "text": (
                f"以下是一段第一人称视角的连续图像序列（共 {len(model_input.frames)} 帧），"
                f"按时间顺序排列，代表一个智能体在室内场景中的移动过程。\n\n"
                f"请仔细观察这段视觉序列，然后回答问题：\n\n"
                f"问题：{model_input.question_text}\n\n"
                f"注意：请只根据图像内容作答，给出简洁明确的方向答案（如：正前方、左后方、正右方等）。"
            )
        }
    ]

    # 按顺序添加所有帧图像
    for i, frame in enumerate(model_input.frames):
        if frame["rgb"]:
            message_content.append({
                "type": "text",
                "text": f"[帧 {i+1}/{len(model_input.frames)}]"
            })
            message_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{frame['rgb']}"
                }
            })

    start_time = time.time()

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": message_content
                }
            ],
            temperature=0.3,
            max_tokens=1000,
            timeout=600  # 600 秒，处理大量图像时需要更长时间
        )

        response_time_ms = int((time.time() - start_time) * 1000)

        return {
            "model_response": response.choices[0].message.content,
            "response_time_ms": response_time_ms,
            "raw_response": {
                "model": response.model,
                "usage": response.usage.model_dump() if response.usage else None
            }
        }

    except Exception as e:
        return {
            "model_response": "",
            "response_time_ms": int((time.time() - start_time) * 1000),
            "error": str(e),
            "raw_response": None
        }


def call_minimax_api_v2(model_input: ModelInputV2) -> Dict[str, Any]:
    """
    调用 MiniMax API（支持多模态）

    Args:
        model_input: ModelInputV2 对象

    Returns:
        响应字典
    """
    import openai

    api_key = os.getenv("MINIMAX_API_KEY")
    base_url = os.getenv("MINIMAX_BASE_URL")
    model_name = os.getenv("MINIMAX_MODEL", "MiniMax-M2.7")

    if not api_key:
        raise ValueError("未找到 MINIMAX_API_KEY")

    client = openai.OpenAI(api_key=api_key, base_url=base_url)

    # 构建消息内容：问题 + 连续帧（模拟视频）
    message_content = [
        {
            "type": "text",
            "text": (
                f"以下是一段第一人称视角的连续图像序列（共 {len(model_input.frames)} 帧），"
                f"按时间顺序排列，代表一个智能体在室内场景中的移动过程。\n\n"
                f"请仔细观察这段视觉序列，然后回答问题：\n\n"
                f"问题：{model_input.question_text}\n\n"
                f"注意：请只根据图像内容作答，给出简洁明确的方向答案（如：正前方、左后方、正右方等）。"
            )
        }
    ]

    for i, frame in enumerate(model_input.frames):
        if frame["rgb"]:
            message_content.append({
                "type": "text",
                "text": f"[帧 {i+1}/{len(model_input.frames)}]"
            })
            message_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{frame['rgb']}"
                }
            })

    start_time = time.time()

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": message_content
                }
            ],
            temperature=0.3,
            max_tokens=1000,
            timeout=600  # 600 秒，处理大量图像时需要更长时间
        )

        response_time_ms = int((time.time() - start_time) * 1000)

        return {
            "model_response": response.choices[0].message.content,
            "response_time_ms": response_time_ms,
            "raw_response": {
                "model": response.model,
                "usage": response.usage.model_dump() if response.usage else None
            }
        }

    except Exception as e:
        return {
            "model_response": "",
            "response_time_ms": int((time.time() - start_time) * 1000),
            "error": str(e),
            "raw_response": None
        }


def call_doubao_api_v2(model_input: ModelInputV2) -> Dict[str, Any]:
    """
    调用豆包 API

    Args:
        model_input: ModelInputV2 对象

    Returns:
        响应字典
    """
    # 豆包暂时返回占位符
    return {
        "model_response": "豆包 API 暂未实现多模态支持",
        "response_time_ms": 0,
        "error": "Not implemented",
        "raw_response": None
    }


# ─────────────────────────────────────────────
# 主评测函数
# ─────────────────────────────────────────────

def evaluate_model_v2(
    exam_path: str,
    model_name: str,
    output_path: str = None
) -> EvaluationResultV2:
    """
    向被测模型发送纯问题 + 多模态证据

    Args:
        exam_path: exam_v2.json 路径
        model_name: 模型名称（kimi/minimax/doubao）
        output_path: 输出路径

    Returns:
        EvaluationResultV2 对象
    """
    # 加载考卷
    with open(exam_path, 'r', encoding='utf-8') as f:
        exam_dict = json.load(f)

    # 重建 ExamPaperV2 对象（简化版，只需要必要字段）
    exam_id = exam_dict["exam_id"]
    questions_data = exam_dict["questions"]
    evidence = exam_dict["multimodal_evidence"]

    print(f"\n[Evaluator V2] 开始评测模型: {model_name}")
    print(f"  Exam ID: {exam_id}")
    print(f"  Questions: {len(questions_data)}")

    # 选择 API 调用函数
    if model_name == "kimi":
        api_func = call_kimi_api_v2
    elif model_name == "minimax":
        api_func = call_minimax_api_v2
    elif model_name == "doubao":
        api_func = call_doubao_api_v2
    else:
        raise ValueError(f"未知模型: {model_name}")

    # 逐题评测
    responses = []
    correct_count = 0

    for i, q_dict in enumerate(questions_data, 1):
        print(f"\n[{i}/{len(questions_data)}] {q_dict['question_id']}")
        print(f"  Question: {q_dict['question_text'][:60]}...")

        # 重建 QuestionV2 对象
        from schema.question_v2 import Pose6DoF
        trajectory_window = [
            Pose6DoF(**pose) for pose in q_dict["trajectory_window"]
        ]
        question = QuestionV2(**{**q_dict, "trajectory_window": trajectory_window})

        # 准备输入
        model_input = prepare_multimodal_input_v2(question, evidence)

        # 调用 API
        api_response = api_func(model_input)

        # 判断正确性
        model_answer = api_response["model_response"].strip()
        ground_truth = question.ground_truth_answer.strip()
        is_correct, score, judge_detail = judge_answer(
            model_answer, ground_truth, question.capability
        )

        if is_correct:
            correct_count += 1
            print(f"  ✅ 正确 (score={score:.2f})")
        else:
            print(f"  ❌ 错误 (score={score:.2f})")
            print(f"     模型回答: {model_answer[:80]}")
            print(f"     标准答案: {ground_truth}")
            print(f"     判定依据: {judge_detail}")

        responses.append({
            "question_id": question.question_id,
            "model_response": model_answer,
            "ground_truth": ground_truth,
            "is_correct": is_correct,
            "score": score,
            "judge_detail": judge_detail,
            "response_time_ms": api_response["response_time_ms"],
            "raw_response": api_response.get("raw_response"),
            "error": api_response.get("error")
        })

    # 计算指标
    accuracy = correct_count / len(questions_data) if questions_data else 0.0
    avg_response_time = sum(r["response_time_ms"] for r in responses) / len(responses) if responses else 0

    # 按能力和难度分组统计
    capability_accuracy = {}
    difficulty_accuracy = {}

    for r, q_dict in zip(responses, questions_data):
        cap = q_dict["capability"]
        diff = q_dict["difficulty"]

        if cap not in capability_accuracy:
            capability_accuracy[cap] = {"correct": 0, "total": 0}
        capability_accuracy[cap]["total"] += 1
        if r["is_correct"]:
            capability_accuracy[cap]["correct"] += 1

        if diff not in difficulty_accuracy:
            difficulty_accuracy[diff] = {"correct": 0, "total": 0}
        difficulty_accuracy[diff]["total"] += 1
        if r["is_correct"]:
            difficulty_accuracy[diff]["correct"] += 1

    # 转换为百分比
    for cap in capability_accuracy:
        stat = capability_accuracy[cap]
        capability_accuracy[cap] = stat["correct"] / stat["total"] if stat["total"] > 0 else 0.0

    for diff in difficulty_accuracy:
        stat = difficulty_accuracy[diff]
        difficulty_accuracy[diff] = stat["correct"] / stat["total"] if stat["total"] > 0 else 0.0

    metrics = {
        "accuracy": accuracy,
        "correct_count": correct_count,
        "total_questions": len(questions_data),
        "avg_response_time_ms": avg_response_time,
        "capability_accuracy": capability_accuracy,
        "difficulty_accuracy": difficulty_accuracy
    }

    result = EvaluationResultV2(
        exam_id=exam_id,
        model_name=model_name,
        responses=responses,
        metrics=metrics,
        timestamp=datetime.now()
    )

    # 保存结果
    if output_path:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        result_dict = result.model_dump(mode='json')

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result_dict, f, indent=2, ensure_ascii=False)

        print(f"\n✅ 评测结果已保存: {output_file}")

    print(f"\n[Summary]")
    print(f"  准确率: {accuracy * 100:.1f}% ({correct_count}/{len(questions_data)})")
    print(f"  平均响应时间: {avg_response_time:.0f}ms")
    print(f"  能力准确率: {capability_accuracy}")
    print(f"  难度准确率: {difficulty_accuracy}")

    return result


def main():
    """主函数：命令行接口"""
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate Runner V2 - 多模态评测")
    parser.add_argument("exam", help="exam_v2.json 路径")
    parser.add_argument("--model", required=True, choices=["kimi", "minimax", "doubao"], help="被测模型")
    parser.add_argument("-o", "--output", help="输出路径")

    args = parser.parse_args()

    output_path = args.output or f"outputs/answers/result_{args.model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    evaluate_model_v2(
        args.exam,
        args.model,
        output_path=output_path
    )


if __name__ == "__main__":
    main()
