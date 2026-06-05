"""
evaluate_runner_v2.py (Enhanced)
---------------------------------
模型测试器 V2 增强版：使用统一的适配器接口

核心改进：
1. 使用 model_adapters.py 统一多模态接口
2. 更清晰的错误处理和重试机制
3. 更完善的评测指标计算
4. 支持批量评测和进度跟踪

数据流：
  exam_v2.json
        ↓
  [本模块] 准备 MultimodalInput
        ↓
  ModelAdapter (Kimi/MiniMax/Doubao)
        ↓
  EvaluationResultV2.json
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent))

from schema.question_v2 import (
    ExamPaperV2,
    QuestionV2,
    EvaluationResultV2,
    Pose6DoF
)

from core.model_adapters import (
    get_adapter,
    MultimodalInput,
    MultimodalFrame,
    PoseData
)


def prepare_multimodal_input(
    question: QuestionV2,
    evidence: Dict,
    data_dir: Path
) -> MultimodalInput:
    """
    从 QuestionV2 和 evidence 准备 MultimodalInput

    Args:
        question: QuestionV2 对象
        evidence: multimodal_evidence 字典
        data_dir: 数据根目录（用于解析相对路径）

    Returns:
        MultimodalInput 对象
    """
    # 准备帧数据
    frames = []
    for frame_id in question.evidence_frame_ids:
        # RGB 路径
        rgb_relative = evidence["rgb_frames"].get(str(frame_id))
        rgb_path = str(data_dir / rgb_relative) if rgb_relative else None

        # Depth 路径
        depth_relative = None
        if evidence.get("depth_frames"):
            depth_relative = evidence["depth_frames"].get(str(frame_id))
        depth_path = str(data_dir / depth_relative) if depth_relative else None

        frames.append(MultimodalFrame(
            frame_id=frame_id,
            rgb_path=rgb_path,
            depth_path=depth_path
        ))

    # 准备轨迹数据
    trajectory = [
        PoseData(
            frame_id=pose.frame_id,
            position=pose.position,
            rotation=pose.rotation,
            euler_angles=pose.euler_angles,
            timestamp=pose.timestamp
        )
        for pose in question.trajectory_window
    ]

    return MultimodalInput(
        question_text=question.question_text,
        frames=frames,
        trajectory=trajectory,
        temperature=0.3,
        max_tokens=1000
    )


def evaluate_answer(model_answer: str, ground_truth: str) -> Dict[str, Any]:
    """
    评估模型答案的正确性

    Args:
        model_answer: 模型回答
        ground_truth: 标准答案

    Returns:
        评估结果字典
    """
    model_answer_clean = model_answer.strip().lower()
    ground_truth_clean = ground_truth.strip().lower()

    # 简单的包含匹配
    is_correct = ground_truth_clean in model_answer_clean

    # 计算相似度（可以后续扩展为语义相似度）
    similarity = 1.0 if is_correct else 0.0

    return {
        "is_correct": is_correct,
        "similarity": similarity,
        "match_method": "substring"
    }


def evaluate_model_v2(
    exam_path: str,
    model_name: str,
    output_path: str = None,
    data_dir: str = None
) -> EvaluationResultV2:
    """
    评测模型（V2 增强版）

    Args:
        exam_path: exam_v2.json 路径
        model_name: 模型名称 (kimi/minimax/doubao)
        output_path: 输出路径
        data_dir: 数据根目录（用于解析图像路径）

    Returns:
        EvaluationResultV2 对象
    """
    # 解析数据目录
    exam_file = Path(exam_path)
    if data_dir is None:
        data_dir = exam_file.parent
    else:
        data_dir = Path(data_dir)

    # 加载考卷
    print(f"\n{'=' * 60}")
    print(f"[Evaluate Runner V2] 模型评测")
    print(f"{'=' * 60}")
    print(f"  模型: {model_name}")
    print(f"  考卷: {exam_path}")
    print(f"  数据目录: {data_dir}")
    print()

    with open(exam_path, 'r', encoding='utf-8') as f:
        exam_dict = json.load(f)

    exam_id = exam_dict["exam_id"]
    questions_data = exam_dict["questions"]
    evidence = exam_dict["multimodal_evidence"]

    print(f"  Exam ID: {exam_id}")
    print(f"  题目数量: {len(questions_data)}")
    print()

    # 获取适配器
    try:
        adapter = get_adapter(model_name)
        print(f"✓ 适配器加载成功: {adapter.model_name}")
    except Exception as e:
        print(f"✗ 适配器加载失败: {e}")
        raise

    # 逐题评测
    responses = []
    correct_count = 0

    for i, q_dict in enumerate(questions_data, 1):
        print(f"\n[{i}/{len(questions_data)}] {q_dict['question_id']}")
        print(f"  问题: {q_dict['question_text'][:70]}...")
        print(f"  能力: {q_dict['capability']}")
        print(f"  难度: {q_dict['difficulty']}")

        # 重建 QuestionV2 对象
        trajectory_window = [Pose6DoF(**pose) for pose in q_dict["trajectory_window"]]
        question = QuestionV2(**{**q_dict, "trajectory_window": trajectory_window})

        # 准备输入
        try:
            model_input = prepare_multimodal_input(question, evidence, data_dir)
            print(f"  证据帧: {len(model_input.frames)}")
        except Exception as e:
            print(f"  ✗ 准备输入失败: {e}")
            responses.append({
                "question_id": question.question_id,
                "model_response": "",
                "ground_truth": question.ground_truth_answer,
                "is_correct": False,
                "response_time_ms": 0,
                "error": f"Input preparation failed: {str(e)}"
            })
            continue

        # 调用模型
        try:
            api_response = adapter(model_input, retry=2)

            if api_response.error:
                print(f"  ✗ API 调用失败: {api_response.error}")
                responses.append({
                    "question_id": question.question_id,
                    "model_response": api_response.answer,
                    "ground_truth": question.ground_truth_answer,
                    "is_correct": False,
                    "response_time_ms": api_response.response_time_ms,
                    "error": api_response.error,
                    "token_usage": api_response.token_usage
                })
                continue

            # 评估答案
            eval_result = evaluate_answer(api_response.answer, question.ground_truth_answer)

            if eval_result["is_correct"]:
                correct_count += 1
                print(f"  ✓ 正确")
            else:
                print(f"  ✗ 错误")
                print(f"     模型: {api_response.answer[:80]}")
                print(f"     标准: {question.ground_truth_answer}")

            print(f"  响应时间: {api_response.response_time_ms}ms")

            responses.append({
                "question_id": question.question_id,
                "model_response": api_response.answer,
                "ground_truth": question.ground_truth_answer,
                "is_correct": eval_result["is_correct"],
                "similarity": eval_result["similarity"],
                "response_time_ms": api_response.response_time_ms,
                "token_usage": api_response.token_usage,
                "raw_response": api_response.raw_response
            })

        except Exception as e:
            print(f"  ✗ 评测失败: {e}")
            import traceback
            traceback.print_exc()
            responses.append({
                "question_id": question.question_id,
                "model_response": "",
                "ground_truth": question.ground_truth_answer,
                "is_correct": False,
                "response_time_ms": 0,
                "error": str(e)
            })

    # 计算指标
    total_questions = len(questions_data)
    accuracy = correct_count / total_questions if total_questions > 0 else 0.0

    valid_responses = [r for r in responses if "error" not in r or r.get("response_time_ms", 0) > 0]
    avg_response_time = (
        sum(r["response_time_ms"] for r in valid_responses) / len(valid_responses)
        if valid_responses else 0
    )

    # 按能力维度统计
    capability_stats = {}
    for r, q_dict in zip(responses, questions_data):
        cap = q_dict["capability"]
        if cap not in capability_stats:
            capability_stats[cap] = {"correct": 0, "total": 0}
        capability_stats[cap]["total"] += 1
        if r.get("is_correct", False):
            capability_stats[cap]["correct"] += 1

    capability_accuracy = {
        cap: stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
        for cap, stats in capability_stats.items()
    }

    # 按难度统计
    difficulty_stats = {}
    for r, q_dict in zip(responses, questions_data):
        diff = q_dict["difficulty"]
        if diff not in difficulty_stats:
            difficulty_stats[diff] = {"correct": 0, "total": 0}
        difficulty_stats[diff]["total"] += 1
        if r.get("is_correct", False):
            difficulty_stats[diff]["correct"] += 1

    difficulty_accuracy = {
        diff: stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
        for diff, stats in difficulty_stats.items()
    }

    # 构建指标
    metrics = {
        "accuracy": accuracy,
        "correct_count": correct_count,
        "total_questions": total_questions,
        "avg_response_time_ms": avg_response_time,
        "capability_accuracy": capability_accuracy,
        "difficulty_accuracy": difficulty_accuracy,
        "capability_stats": capability_stats,
        "difficulty_stats": difficulty_stats
    }

    # 创建结果对象
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

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result.model_dump(mode='json'), f, indent=2, ensure_ascii=False)

        print(f"\n✓ 评测结果已保存: {output_file}")

    # 打印总结
    print(f"\n{'=' * 60}")
    print("[评测总结]")
    print(f"{'=' * 60}")
    print(f"  准确率: {accuracy * 100:.1f}% ({correct_count}/{total_questions})")
    print(f"  平均响应时间: {avg_response_time:.0f}ms")
    print(f"\n  能力准确率:")
    for cap, acc in capability_accuracy.items():
        stats = capability_stats[cap]
        print(f"    - {cap}: {acc * 100:.1f}% ({stats['correct']}/{stats['total']})")
    print(f"\n  难度准确率:")
    for diff, acc in difficulty_accuracy.items():
        stats = difficulty_stats[diff]
        print(f"    - {diff}: {acc * 100:.1f}% ({stats['correct']}/{stats['total']})")
    print(f"{'=' * 60}\n")

    return result


def main():
    """命令行接口"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate Runner V2 - 多模态模型评测器（增强版）"
    )

    parser.add_argument(
        "exam",
        help="exam_v2.json 文件路径"
    )

    parser.add_argument(
        "--model",
        required=True,
        choices=["kimi", "minimax", "doubao"],
        help="被测模型名称"
    )

    parser.add_argument(
        "-o", "--output",
        help="输出文件路径（默认：outputs/answers/result_{model}_{timestamp}.json）"
    )

    parser.add_argument(
        "--data-dir",
        help="数据根目录（用于解析图像相对路径，默认使用 exam 文件所在目录）"
    )

    args = parser.parse_args()

    # 默认输出路径
    if not args.output:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"outputs/answers/result_{args.model}_{timestamp}.json"

    # 运行评测
    evaluate_model_v2(
        exam_path=args.exam,
        model_name=args.model,
        output_path=args.output,
        data_dir=args.data_dir
    )


if __name__ == "__main__":
    main()
