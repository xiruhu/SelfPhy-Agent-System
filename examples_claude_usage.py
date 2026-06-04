"""
Claude API 使用示例
演示如何在 SelfPhy-Agent-System 中使用 Claude 进行空间推理评测
"""

import sys
from pathlib import Path
from dotenv import load_dotenv

# 设置 UTF-8 输出
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 加载环境变量
load_dotenv()

from core.claude_adapter import ClaudeAdapter
from core.evaluate_runner import EvaluationRunner


def example_1_simple_question():
    """示例 1: 简单的空间推理问题"""
    print("\n" + "=" * 80)
    print("示例 1: 简单的空间推理问题")
    print("=" * 80 + "\n")

    adapter = ClaudeAdapter()

    prompt = """
你在一个房间里，面向北方。你看到：
- 前方 3 米处有一把椅子
- 右侧 2 米处有一张桌子

现在你向右转 90 度，然后向前走 2 米。

问题：椅子现在在你的哪个方向？距离你多远？

请用 JSON 格式回答：
{
  "direction": "方向（前/后/左/右）",
  "distance": "距离（米）",
  "reasoning": "推理过程"
}
"""

    result = adapter.call(prompt=prompt, images=[], temperature=0.3)

    print("Claude 的回答：")
    print(result['response'])
    print(f"\nToken 消耗: {result['token_count']}")


def example_2_multimodal_question():
    """示例 2: 多模态问题（图片 + 文本）"""
    print("\n" + "=" * 80)
    print("示例 2: 多模态空间推理问题")
    print("=" * 80 + "\n")

    # 查找示例图片
    sample_images = list(Path("data/raw/habitat").rglob("frame_*.jpg"))

    if not sample_images:
        print("⚠ 未找到示例图片，请先运行: python create_sample_data.py")
        return

    # 选择 3 张图片
    test_images = [str(img) for img in sample_images[:3]]

    adapter = ClaudeAdapter()

    prompt = """
这是一个第一人称视角的连续帧序列。

请回答：
1. 你观察到了什么物体？
2. 从第一帧到最后一帧，你认为相机做了什么运动？（前进/后退/转向？）
3. 如果第一帧中有一个红色方块在你的右侧，现在它应该在哪里？

请用中文简短回答（3-5 句话）。
"""

    result = adapter.call(prompt=prompt, images=test_images, temperature=0.3)

    print(f"测试图片: {len(test_images)} 张")
    for i, img in enumerate(test_images):
        print(f"  Frame {i}: {img}")

    print("\nClaude 的回答：")
    print(result['response'])
    print(f"\nToken 消耗: {result['token_count']}")


def example_3_batch_evaluation():
    """示例 3: 批量评测"""
    print("\n" + "=" * 80)
    print("示例 3: 批量评测多个问题")
    print("=" * 80 + "\n")

    # 检查是否有测试问题
    exam_files = list(Path("outputs/exams").glob("exam_*.json"))

    if not exam_files:
        print("⚠ 未找到测试问题文件")
        print("   请先运行: python run_pipeline.py")
        return

    runner = EvaluationRunner()

    # 使用第一个测试文件
    exam_file = exam_files[0]
    print(f"使用测试文件: {exam_file}\n")

    # 运行评测
    responses = runner.run_evaluation(
        questions_path=str(exam_file),
        model_name="claude-sonnet-4-6"
    )

    # 保存结果
    output_path = runner.save_responses(responses)

    # 统计
    total = len(responses)
    errors = sum(1 for r in responses if "error" in r.parsed_answer)
    success = total - errors

    print("\n" + "=" * 80)
    print("评测结果统计")
    print("=" * 80)
    print(f"总问题数: {total}")
    print(f"成功: {success}")
    print(f"失败: {errors}")
    print(f"成功率: {success/total*100:.1f}%")
    print(f"\n结果已保存到: {output_path}")


def example_4_error_analysis():
    """示例 4: 错误分析（演示如何使用 Claude 进行反思）"""
    print("\n" + "=" * 80)
    print("示例 4: 错误分析与反思")
    print("=" * 80 + "\n")

    adapter = ClaudeAdapter()

    # 模拟一个错误的回答
    question = "在视频结束时，椅子相对于你的位置在哪里？"
    ground_truth = "在我的左后方约 2 米处"
    model_answer = "在我的右前方"

    # 让 Claude 分析错误原因
    prompt = f"""
你是一个空间推理错误分析专家。

**问题**: {question}
**正确答案**: {ground_truth}
**模型回答**: {model_answer}

请分析这个错误可能的原因，从以下角度：
1. **物理及语义对齐**: 模型是否理解了问题的物理含义？
2. **空间位置重塑**: 模型是否正确追踪了空间变换？
3. **视场边界校验**: 物体是否在视野范围内？
4. **根因分类**: 这是什么类型的错误？（空间记忆衰减/遮挡盲区/坐标计算错误）

请用 JSON 格式回答：
{{
  "error_type": "错误类型",
  "root_cause": "根本原因",
  "analysis": "详细分析",
  "confidence": 0.0-1.0
}}
"""

    result = adapter.call(prompt=prompt, images=[], temperature=0.3)

    print("Claude 的错误分析：")
    print(result['response'])
    print(f"\nToken 消耗: {result['token_count']}")


def main():
    """运行所有示例"""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "Claude API 使用示例" + " " * 37 + "║")
    print("╚" + "=" * 78 + "╝")

    try:
        # 运行示例
        example_1_simple_question()
        example_2_multimodal_question()
        example_3_batch_evaluation()
        example_4_error_analysis()

        print("\n" + "=" * 80)
        print("✓ 所有示例运行完成!")
        print("=" * 80 + "\n")

    except Exception as e:
        print(f"\n✗ 示例运行失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
