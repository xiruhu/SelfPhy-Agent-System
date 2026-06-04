"""
测试 Claude 模型在完整流程中的调用
"""

import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv
from core.claude_adapter import ClaudeAdapter

# 设置 UTF-8 输出（Windows 兼容）
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 加载环境变量
load_dotenv()


def test_claude_simple():
    """测试 1: 简单文本调用"""
    print("=" * 80)
    print("测试 1: 简单文本调用")
    print("=" * 80)

    adapter = ClaudeAdapter()

    result = adapter.call(
        prompt="请用中文回答：你是什么模型？",
        images=[],
        temperature=0.3
    )

    print(f"✓ 响应: {result['response']}")
    print(f"✓ Token 数: {result['token_count']}")
    print()


def test_claude_with_image():
    """测试 2: 多模态调用（图片+文本）"""
    print("=" * 80)
    print("测试 2: 多模态调用（图片+文本）")
    print("=" * 80)

    # 查找示例图片
    sample_images = list(Path("data/raw/habitat").rglob("frame_*.jpg"))

    if not sample_images:
        print("⚠ 未找到示例图片，跳过此测试")
        print("   请先运行: python create_sample_data.py")
        return

    # 取前 2 张图片
    test_images = [str(img) for img in sample_images[:2]]

    adapter = ClaudeAdapter()

    prompt = """
请观察这些图片，并回答：
1. 你看到了什么？
2. 这些图片之间有什么变化？

请用中文简短回答（2-3 句话）。
"""

    result = adapter.call(
        prompt=prompt,
        images=test_images,
        temperature=0.3
    )

    print(f"✓ 测试图片: {len(test_images)} 张")
    for img in test_images:
        print(f"  - {img}")
    print(f"\n✓ 响应:\n{result['response']}")
    print(f"\n✓ Token 数: {result['token_count']}")
    print()


def test_claude_spatial_reasoning():
    """测试 3: 空间推理能力测试"""
    print("=" * 80)
    print("测试 3: 空间推理能力测试")
    print("=" * 80)

    adapter = ClaudeAdapter()

    prompt = """
假设你在一个房间里，初始朝向北方。然后你执行以下动作：
1. 向前走 3 米
2. 向右转 90 度
3. 向前走 2 米
4. 向左转 90 度
5. 向前走 1 米

请回答：
1. 你现在相对于起点的位置是什么？（用坐标表示，起点为 (0, 0)，北为 +Y，东为 +X）
2. 你现在的朝向是什么？

请用 JSON 格式回答：
{
  "position": [x, y],
  "direction": "方向"
}
"""

    result = adapter.call(
        prompt=prompt,
        images=[],
        temperature=0.3
    )

    print(f"✓ 响应:\n{result['response']}")
    print(f"\n✓ Token 数: {result['token_count']}")
    print()


def test_evaluate_runner_integration():
    """测试 4: 集成到 EvaluationRunner"""
    print("=" * 80)
    print("测试 4: 集成到 EvaluationRunner")
    print("=" * 80)

    from core.evaluate_runner import EvaluationRunner

    runner = EvaluationRunner()

    # 检查 Claude 是否已配置
    if "claude-sonnet-4-6" in runner.adapters:
        print("✓ Claude Sonnet 4.6 已配置")
    else:
        print("✗ Claude Sonnet 4.6 未配置")
        return

    # 检查是否有测试问题
    exam_files = list(Path("outputs/exams").glob("exam_*.json"))

    if not exam_files:
        print("⚠ 未找到测试问题文件")
        print("   请先运行: python run_pipeline.py")
        return

    # 使用第一个测试文件
    exam_file = exam_files[0]
    print(f"✓ 使用测试文件: {exam_file}")

    # 只测试第一个问题
    with open(exam_file, 'r', encoding='utf-8') as f:
        questions = json.load(f)

    if not questions:
        print("✗ 测试文件为空")
        return

    # 只测试第一题
    test_question = questions[0]
    print(f"✓ 测试问题: {test_question['question_id']}")

    # 保存为临时文件
    temp_exam = Path("outputs/exams/temp_test.json")
    with open(temp_exam, 'w', encoding='utf-8') as f:
        json.dump([test_question], f, indent=2, ensure_ascii=False)

    try:
        # 运行评测
        responses = runner.run_evaluation(
            questions_path=str(temp_exam),
            model_name="claude-sonnet-4-6"
        )

        if responses:
            print(f"\n✓ 评测成功!")
            print(f"  响应: {responses[0].raw_response[:200]}...")
            print(f"  Token 数: {responses[0].token_count}")
        else:
            print("✗ 评测失败：无响应")

    except Exception as e:
        print(f"✗ 评测失败: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # 清理临时文件
        if temp_exam.exists():
            temp_exam.unlink()

    print()


def main():
    """运行所有测试"""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "Claude API 调用测试套件" + " " * 34 + "║")
    print("╚" + "=" * 78 + "╝")
    print()

    # 检查 API Key
    api_key = os.getenv("ANTHROPIC_API_KEY")
    base_url = os.getenv("ANTHROPIC_BASE_URL")

    if not api_key:
        print("✗ 错误: 未找到 ANTHROPIC_API_KEY")
        print("  请在 .env 文件中配置 API Key")
        return

    print(f"✓ API Key: {api_key[:20]}...")
    print(f"✓ Base URL: {base_url or '默认'}")
    print()

    try:
        # 运行测试
        test_claude_simple()
        test_claude_with_image()
        test_claude_spatial_reasoning()
        test_evaluate_runner_integration()

        print("=" * 80)
        print("✓ 所有测试完成!")
        print("=" * 80)

    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
