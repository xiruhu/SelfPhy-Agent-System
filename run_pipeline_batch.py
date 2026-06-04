"""
批量运行脚本 - 处理多个 episode
运行完整的 SelfPhy-Agent 评测流程（支持多 episode）
"""

import sys
import os
from pathlib import Path
from datetime import datetime
import json

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent))

from core.cv2_processor import DataPerceptionModule
from core.prompt_generator import ExamGenerator
from core.evaluate_runner import EvaluationRunner
from core.gpt4o_reflector import InspectorAgent
from core.rag_manager import RAGManager
from core.analytics_viz import AnalyticsVisualizer


def print_banner(text: str):
    """打印横幅"""
    print("\n" + "=" * 80)
    print(f"  {text}")
    print("=" * 80 + "\n")


def process_single_episode(episode_id: str, model_name: str, runner: EvaluationRunner):
    """
    处理单个 episode

    Args:
        episode_id: episode 标识符
        model_name: 模型名称
        runner: 评测运行器

    Returns:
        (trajectory_path, questions_path, responses_path, analyses_path)
    """
    print_banner(f"处理 Episode: {episode_id}")

    # Step 1: 数据感知层
    print(f"\n[Step 1] 提取关键帧...")
    processor = DataPerceptionModule()

    try:
        trajectory = processor.process_episode(
            episode_id=episode_id,
            rotation_threshold=30.0,
            translation_threshold=1.0
        )

        processor.visualize_trajectory(trajectory)
        trajectory_path = f"outputs/trajectories/{episode_id}.json"
        print(f"✓ 关键帧提取完成: {len(trajectory.keyframes)} 帧")
    except Exception as e:
        print(f"✗ 数据处理失败: {e}")
        return None, None, None, None

    # Step 2: 考试生成器
    print(f"\n[Step 2] 生成考题...")
    generator = ExamGenerator()

    try:
        questions = generator.generate_questions(
            trajectory_path=trajectory_path,
            num_questions=10,
            enable_adversarial=False
        )

        questions_path = generator.save_questions(
            questions,
            f"exam_{episode_id}.json"
        )
        print(f"✓ 生成考题: {len(questions)} 道")
    except Exception as e:
        print(f"✗ 考题生成失败: {e}")
        return trajectory_path, None, None, None

    # Step 3: 测试执行器
    print(f"\n[Step 3] 运行评测...")

    try:
        responses = runner.run_evaluation(
            questions_path=questions_path,
            model_name=model_name
        )

        responses_path = runner.save_responses(
            responses,
            f"responses_{model_name}_{episode_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )

        total = len(responses)
        errors = sum(1 for r in responses if "error" in r.parsed_answer)
        print(f"✓ 评测完成: {total} 道题, 成功: {total - errors}, 失败: {errors}")
    except Exception as e:
        print(f"✗ 评测失败: {e}")
        return trajectory_path, questions_path, None, None

    # Step 4: 错题分析器
    print(f"\n[Step 4] 错误分析...")
    inspector = InspectorAgent()

    try:
        analyses = inspector.analyze_errors(
            responses_path=responses_path,
            questions_path=questions_path,
            trajectory_path=trajectory_path
        )

        if analyses:
            analyses_path = inspector.save_analyses(
                analyses,
                f"error_analysis_{episode_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            print(f"✓ 错误分析完成: {len(analyses)} 个错误")
        else:
            print("✓ 所有答案正确，无需分析")
            analyses_path = None
    except Exception as e:
        print(f"✗ 错误分析失败: {e}")
        analyses_path = None

    return trajectory_path, questions_path, responses_path, analyses_path


def main():
    """主函数：批量运行评测流程"""

    print_banner("SelfPhy-Agent-System - 批量评测流程")

    start_time = datetime.now()

    # Step 0: 初始化 RAG 知识库
    print_banner("Step 0: 初始化 RAG 知识库")

    try:
        rag = RAGManager()
        rag.initialize_default_rules()
        rag.export_rules("outputs/knowledge_base/physical_rules.json")
        print("✓ RAG 知识库初始化完成")
    except Exception as e:
        print(f"✗ RAG 初始化失败: {e}")

    # 检查 Habitat 数据
    habitat_data_dir = Path("data/raw/habitat")
    if not habitat_data_dir.exists() or not list(habitat_data_dir.glob("episode_*")):
        print("\n⚠ 未找到 Habitat 数据")
        print("请先运行: python create_sample_data.py")
        return

    # 获取所有 episode
    episodes = sorted(habitat_data_dir.glob("episode_*"))
    print(f"\n找到 {len(episodes)} 个 episode")

    # 初始化评测运行器
    runner = EvaluationRunner()

    if not runner.adapters:
        print("\n⚠ 未配置模型 API 密钥")
        print("请在 .env 文件中配置 ANTHROPIC_API_KEY 或 OPENAI_API_KEY")
        return

    # 选择模型
    if "claude-sonnet-4-6" in runner.adapters:
        model_name = "claude-sonnet-4-6"
    else:
        model_name = list(runner.adapters.keys())[0]

    print(f"使用模型: {model_name}")

    # 处理所有 episode
    all_results = []

    for episode_path in episodes:
        episode_id = episode_path.name

        result = process_single_episode(episode_id, model_name, runner)
        all_results.append({
            "episode_id": episode_id,
            "trajectory_path": result[0],
            "questions_path": result[1],
            "responses_path": result[2],
            "analyses_path": result[3]
        })

    # 生成汇总报告
    print_banner("生成汇总报告")

    summary_path = Path("outputs/reports") / f"batch_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump({
            "model_name": model_name,
            "total_episodes": len(episodes),
            "timestamp": datetime.now().isoformat(),
            "results": all_results
        }, f, indent=2, ensure_ascii=False)

    print(f"✓ 汇总报告已保存: {summary_path}")

    # 完成
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    print_banner("批量评测流程完成")

    print(f"总耗时: {duration:.1f} 秒")
    print(f"处理 episode 数: {len(episodes)}")
    print(f"汇总报告: {summary_path}")

    print("\n下一步:")
    print("  运行 Streamlit 看板: streamlit run app.py")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n用户中断执行")
    except Exception as e:
        print(f"\n\n执行失败: {e}")
        import traceback
        traceback.print_exc()
