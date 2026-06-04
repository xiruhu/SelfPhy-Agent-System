"""
主运行脚本 - 端到端评测流程
运行完整的 SelfPhy-Agent 评测流程
"""

import sys
import os
from pathlib import Path
from datetime import datetime

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent))

from core.cv2_processor import DataPerceptionModule
from core.prompt_generator import ExamGenerator
from core.evaluate_runner import EvaluationRunner
from core.gpt4o_reflector import InspectorAgent
from core.rag_manager import RAGManager
from core.analytics_viz import AnalyticsVisualizer
import json


def print_banner(text: str):
    """打印横幅"""
    print("\n" + "=" * 80)
    print(f"  {text}")
    print("=" * 80 + "\n")


def main():
    """主函数：运行完整评测流程"""

    print_banner("SelfPhy-Agent-System - 自动化评测流程")

    start_time = datetime.now()

    # ========================================================================
    # Step 0: 初始化 RAG 知识库
    # ========================================================================
    print_banner("Step 0: 初始化 RAG 知识库")

    try:
        rag = RAGManager()
        rag.initialize_default_rules()
        rag.export_rules("outputs/knowledge_base/physical_rules.json")
        print("✓ RAG 知识库初始化完成")
    except Exception as e:
        print(f"✗ RAG 初始化失败: {e}")
        print("继续执行后续步骤...")

    # ========================================================================
    # Step 1: 数据感知层 - 提取关键帧
    # ========================================================================
    print_banner("Step 1: 数据感知层 - 提取关键帧")

    processor = DataPerceptionModule()

    # 检查是否有 Habitat 数据
    habitat_data_dir = Path("data/raw/habitat")
    if not habitat_data_dir.exists() or not list(habitat_data_dir.glob("episode_*")):
        print("⚠ 未找到 Habitat 数据")
        print("请将 Habitat episode 数据放置在 data/raw/habitat/ 目录下")
        print("\n目录结构示例:")
        print("data/raw/habitat/")
        print("├── episode_001/")
        print("│   ├── frames/ 或 video.mp4")
        print("│   └── poses.json")
        print("\n跳过数据处理步骤...")
        trajectory_path = None
    else:
        # 处理第一个 episode
        episodes = sorted(habitat_data_dir.glob("episode_*"))
        episode_id = episodes[0].name

        print(f"处理 episode: {episode_id}")

        try:
            trajectory = processor.process_episode(
                episode_id=episode_id,
                rotation_threshold=30.0,
                translation_threshold=1.0
            )

            # 可视化轨迹
            processor.visualize_trajectory(trajectory)

            trajectory_path = f"outputs/trajectories/{episode_id}.json"
            print(f"✓ 关键帧提取完成: {len(trajectory.keyframes)} 帧")

        except Exception as e:
            print(f"✗ 数据处理失败: {e}")
            import traceback
            traceback.print_exc()
            return

    # ========================================================================
    # Step 2: 考试生成器 - 自动出题
    # ========================================================================
    print_banner("Step 2: 考试生成器 - 自动出题")

    if trajectory_path and Path(trajectory_path).exists():
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

            # 显示题型分布
            from collections import Counter
            type_dist = Counter([q.question_type.value for q in questions])
            print("\n题型分布:")
            for qtype, count in type_dist.items():
                print(f"  - {qtype}: {count}")

        except Exception as e:
            print(f"✗ 考题生成失败: {e}")
            import traceback
            traceback.print_exc()
            return
    else:
        print("⚠ 跳过考题生成（无轨迹数据）")
        questions_path = None

    # ========================================================================
    # Step 3: 测试执行器 - 运行评测
    # ========================================================================
    print_banner("Step 3: 测试执行器 - 运行评测")

    if questions_path and Path(questions_path).exists():
        runner = EvaluationRunner()

        # 检查是否配置了 API 密钥
        if not runner.adapters:
            print("⚠ 未配置模型 API 密钥")
            print("请在 .env 文件中配置 ANTHROPIC_API_KEY 或 OPENAI_API_KEY 等密钥")
            print("跳过模型评测...")
            responses_path = None
        else:
            # 优先使用 Claude，否则使用第一个可用的模型
            if "claude-sonnet-4-6" in runner.adapters:
                model_name = "claude-sonnet-4-6"
            else:
                model_name = list(runner.adapters.keys())[0]
            print(f"使用模型: {model_name}")

            try:
                responses = runner.run_evaluation(
                    questions_path=questions_path,
                    model_name=model_name
                )

                responses_path = runner.save_responses(responses)

                # 统计
                total = len(responses)
                errors = sum(1 for r in responses if "error" in r.parsed_answer)
                print(f"\n✓ 评测完成: {total} 道题")
                print(f"  - 成功: {total - errors}")
                print(f"  - 失败: {errors}")

            except Exception as e:
                print(f"✗ 评测失败: {e}")
                import traceback
                traceback.print_exc()
                responses_path = None
    else:
        print("⚠ 跳过模型评测（无考题数据）")
        responses_path = None

    # ========================================================================
    # Step 4: 错题分析器 - 四步排除法
    # ========================================================================
    print_banner("Step 4: 错题分析器 - 四步排除法")

    if responses_path and Path(responses_path).exists():
        inspector = InspectorAgent()

        try:
            analyses = inspector.analyze_errors(
                responses_path=responses_path,
                questions_path=questions_path,
                trajectory_path=trajectory_path
            )

            if analyses:
                analyses_path = inspector.save_analyses(analyses)

                # 统计错误类型
                from collections import Counter
                error_types = Counter([a.error_type.value for a in analyses])

                print(f"\n✓ 错误分析完成: {len(analyses)} 个错误")
                print("\n错误类型分布:")
                for error_type, count in error_types.items():
                    print(f"  - {error_type}: {count}")
            else:
                print("✓ 所有答案正确，无需分析")
                analyses_path = None

        except Exception as e:
            print(f"✗ 错误分析失败: {e}")
            import traceback
            traceback.print_exc()
            analyses_path = None
    else:
        print("⚠ 跳过错误分析（无评测结果）")
        analyses_path = None

    # ========================================================================
    # Step 5: 可视化层 - 生成报告
    # ========================================================================
    print_banner("Step 5: 可视化层 - 生成报告")

    # 即使没有错误分析，也尝试生成基础报告
    if responses_path and Path(responses_path).exists():
        visualizer = AnalyticsVisualizer()

        try:
            if analyses_path and Path(analyses_path).exists():
                # 有错误分析数据，生成完整报告
                # 生成雷达图
                visualizer.generate_radar_chart(
                    {model_name: analyses_path},
                    "radar_chart.png"
                )

                # 生成遗忘曲线
                visualizer.generate_forgetting_curve(
                    analyses_path,
                    questions_path,
                    "forgetting_curve.png"
                )

                # 生成热力图
                visualizer.generate_heatmap(
                    analyses_path,
                    questions_path,
                    "heatmap.png"
                )

                # 生成综合报告
                visualizer.generate_summary_report(
                    {model_name: analyses_path},
                    questions_path,
                    "summary_report.png"
                )

                print("✓ 可视化报告生成完成")
                print(f"\n报告位置: outputs/visualizations/")
            else:
                # 没有错误分析，生成基础统计报告
                print("⚠ 无错误分析数据，生成基础统计报告")

                # 生成简单的统计报告
                with open(responses_path, 'r', encoding='utf-8') as f:
                    responses = json.load(f)

                total = len(responses)
                errors = sum(1 for r in responses if "error" in r.get('parsed_answer', {}))
                success = total - errors

                report_path = Path("outputs/reports") / f"basic_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                report_path.parent.mkdir(parents=True, exist_ok=True)

                with open(report_path, 'w', encoding='utf-8') as f:
                    f.write("=" * 80 + "\n")
                    f.write("SelfPhy-Agent 评测基础报告\n")
                    f.write("=" * 80 + "\n\n")
                    f.write(f"模型: {model_name}\n")
                    f.write(f"总题数: {total}\n")
                    f.write(f"成功: {success} ({success/total*100:.1f}%)\n")
                    f.write(f"失败: {errors} ({errors/total*100:.1f}%)\n\n")
                    f.write("失败详情:\n")
                    for r in responses:
                        if "error" in r.get('parsed_answer', {}):
                            f.write(f"  - {r['question_id']}: {r['parsed_answer'].get('error', 'Unknown error')}\n")

                print(f"✓ 基础报告已生成: {report_path}")

        except Exception as e:
            print(f"✗ 报告生成失败: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("⚠ 跳过报告生成（无评测结果）")

    # ========================================================================
    # 完成
    # ========================================================================
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    print_banner("评测流程完成")

    print(f"总耗时: {duration:.1f} 秒")
    print("\n生成的文件:")
    print(f"  - 轨迹数据: {trajectory_path if trajectory_path else 'N/A'}")
    print(f"  - 考题: {questions_path if questions_path else 'N/A'}")
    print(f"  - 评测结果: {responses_path if responses_path else 'N/A'}")
    print(f"  - 错误分析: {analyses_path if analyses_path else 'N/A'}")
    print(f"  - 可视化报告: outputs/visualizations/")

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
